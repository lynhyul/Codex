import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_stratified(labels: List[int], ratio: float, seed: int):
    rng = random.Random(seed)
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(int(y), []).append(i)
    tr, va = [], []
    for _, idxs in by.items():
        rng.shuffle(idxs)
        nv = max(1, int(len(idxs) * ratio))
        va.extend(idxs[:nv])
        tr.extend(idxs[nv:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return torch.tensor(tr), torch.tensor(va)


def make_tf(size: int):
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(model_name: str, nc: int):
    if model_name.startswith("timm:"):
        import timm

        return timm.create_model(model_name.split(":", 1)[1], pretrained=False, num_classes=nc)
    if model_name == "resnet18":
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, nc)
        return m
    if model_name == "resnet34":
        m = models.resnet34(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, nc)
        return m
    if model_name == "resnet50":
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, nc)
        return m
    if model_name == "efficientnet_b0" and hasattr(models, "efficientnet_b0"):
        m = models.efficientnet_b0(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
        return m
    if model_name == "efficientnet_b3" and hasattr(models, "efficientnet_b3"):
        m = models.efficientnet_b3(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
        return m
    return None


def parse_tokens(name: str):
    import re

    a = re.search(r"ADI(\d+)", name)
    b = re.search(r"AOI(\d+)", name)
    c = re.search(r"VRS(\d+)", name)
    d = re.search(r"_([A-Z])_AOI", name)
    return (
        a.group(1) if a else "NA",
        b.group(1) if b else "NA",
        c.group(1) if c else "NA",
        d.group(1) if d else "NA",
    )


def fit_token_map(names: List[str]):
    vals = [set() for _ in range(4)]
    for n in names:
        t = parse_tokens(n)
        for i in range(4):
            vals[i].add(t[i])
    maps = []
    for i in range(4):
        ss = sorted(vals[i])
        maps.append({k: j for j, k in enumerate(ss)})
    return maps


def encode_tokens(names: List[str], maps):
    dims = [len(m) + 1 for m in maps]
    off = [0]
    for d in dims[:-1]:
        off.append(off[-1] + d)
    X = torch.zeros((len(names), sum(dims)))
    for i, n in enumerate(names):
        t = parse_tokens(n)
        for j in range(4):
            idx = maps[j].get(t[j], len(maps[j]))
            X[i, off[j] + idx] = 1.0
    return X


def infer_or_load(cp: Path, split: str, data_dir: Path, cache_dir: Path, device, batch_size, num_workers):
    cache = cache_dir / f"{split}_{cp.stem}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu")
        return blob

    ckpt = torch.load(cp, map_location="cpu")
    cfg = ckpt.get("config", {})
    mn = cfg.get("model_name")
    if not mn:
        return None
    model = build_model(mn, 26)
    if model is None:
        return None

    size = int(cfg.get("img_size", 224))
    tta = bool(cfg.get("tta", False))
    ds = datasets.ImageFolder(str(data_dir), transform=make_tf(size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(mn, len(ds.classes))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    logits = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            lg = model(x)
            if tta:
                lg = (lg + model(torch.flip(x, dims=[3]))) / 2.0
            logits.append(lg.cpu())
            labels.append(y)
    blob = {
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "names": [Path(p).name for p, _ in ds.samples],
        "classes": ds.classes,
        "model_name": mn,
        "img_size": size,
        "tta": tta,
        "val_acc": float(ckpt.get("val_acc", 0.0)),
    }
    torch.save(blob, cache)
    return blob


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_cls(name, x_tr, y_tr, x_va, y_va, x_te, y_te, epochs, lr, device):
    model = MLP(x_tr.size(1), int(y_tr.max().item() + 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds_tr = TensorDataset(x_tr, y_tr)
    ds_va = TensorDataset(x_va, y_va)
    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False)

    best = {"acc": 0.0, "state": None, "epoch": 0}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        outs, ys = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                outs.append(model(xb).cpu())
                ys.append(yb)
        lg = torch.cat(outs)
        yy = torch.cat(ys)
        acc = (lg.argmax(1) == yy).float().mean().item()
        print(f"[{name}] ep {ep:02d} val={acc:.4f}")
        if acc > best["acc"]:
            best = {"acc": acc, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": ep}

    model.load_state_dict(best["state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        te = model(x_te.to(device)).cpu()
    te_acc = (te.argmax(1) == y_te).float().mean().item()
    return model, best, te_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="E:/samsung/Dataset/train")
    ap.add_argument("--test_dir", default="E:/samsung/Dataset/test")
    ap.add_argument("--runs_dir", default="E:/samsung/Dataset/runs")
    ap.add_argument("--out_dir", default="E:/samsung/Dataset/runs/no_filename_eval_all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    ckpts = sorted(Path(args.runs_dir).glob("best_*.pt"))
    valid = []
    for cp in ckpts:
        if cp.stat().st_size <= 0:
            continue
        if "custom" in cp.stem:
            continue
        valid.append(cp)
    print("num checkpoints", len(valid))

    train_blobs = []
    test_blobs = []
    for cp in valid:
        print("infer", cp.name)
        bt = infer_or_load(cp, "train", Path(args.train_dir), cache, device, args.batch_size, args.num_workers)
        be = infer_or_load(cp, "test", Path(args.test_dir), cache, device, args.batch_size, args.num_workers)
        if bt is None or be is None:
            print("skip", cp.name)
            continue
        train_blobs.append((cp.stem, bt))
        test_blobs.append((cp.stem, be))

    names = [n for n, _ in train_blobs]
    # checks
    tr_names = train_blobs[0][1]["names"]
    te_names = test_blobs[0][1]["names"]
    y_tr = train_blobs[0][1]["labels"].long()
    y_te = test_blobs[0][1]["labels"].long()
    for _, b in train_blobs[1:]:
        assert tr_names == b["names"]
        assert torch.equal(y_tr, b["labels"].long())
    for _, b in test_blobs[1:]:
        assert te_names == b["names"]
        assert torch.equal(y_te, b["labels"].long())

    lg_tr = torch.stack([b["logits"] for _, b in train_blobs], dim=1)  # N,M,C
    lg_te = torch.stack([b["logits"] for _, b in test_blobs], dim=1)

    tr_idx, va_idx = split_stratified(y_tr.tolist(), 0.2, args.seed)

    results = []
    for i, (nm, _) in enumerate(train_blobs):
        acc = (lg_te[:, i, :].argmax(1) == y_te).float().mean().item()
        results.append({"name": nm, "type": "expert", "filename_required": False, "test_acc": acc})

    # uniform all
    uni_all_te = lg_te.mean(1)
    uni_all_acc = (uni_all_te.argmax(1) == y_te).float().mean().item()
    uni_all_va = (lg_tr[va_idx].mean(1).argmax(1) == y_tr[va_idx]).float().mean().item()
    results.append({"name": "uniform_all", "type": "ensemble", "filename_required": False, "test_acc": uni_all_acc, "val_acc": uni_all_va})

    # subset search by val for uniform averaging
    m = lg_tr.size(1)
    best_subset = None
    best_va = -1
    for r in range(2, m + 1):
        for comb in itertools.combinations(range(m), r):
            va_acc = (lg_tr[va_idx][:, comb, :].mean(1).argmax(1) == y_tr[va_idx]).float().mean().item()
            if va_acc > best_va:
                best_va = va_acc
                best_subset = comb
    sub_te = (lg_te[:, best_subset, :].mean(1).argmax(1) == y_te).float().mean().item()
    results.append({"name": "uniform_subset_val_best", "type": "ensemble", "filename_required": False, "test_acc": sub_te, "val_acc": best_va, "subset": [names[i] for i in best_subset]})

    # scalar weights
    w = nn.Parameter(torch.zeros(m))
    opt = torch.optim.Adam([w], lr=0.05)
    best_w = None
    best_w_va = -1
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        ww = torch.softmax(w, dim=0)
        logits_train_weighted = (lg_tr[tr_idx] * ww.view(1, -1, 1)).sum(1)
        loss = F.cross_entropy(logits_train_weighted, y_tr[tr_idx])
        loss.backward()
        opt.step()
        with torch.no_grad():
            va = ( (lg_tr[va_idx] * torch.softmax(w, dim=0).view(1,-1,1)).sum(1).argmax(1) == y_tr[va_idx]).float().mean().item()
            if va > best_w_va:
                best_w_va = va
                best_w = w.detach().clone()
    ww = torch.softmax(best_w, dim=0)
    sw_te = ((lg_te * ww.view(1, -1, 1)).sum(1).argmax(1) == y_te).float().mean().item()
    results.append({"name": "scalar_weighted_all", "type": "ensemble", "filename_required": False, "test_acc": sw_te, "val_acc": best_w_va, "weights": ww.tolist()})

    # image-only stacker
    x_img_tr = lg_tr.reshape(lg_tr.size(0), -1)
    x_img_te = lg_te.reshape(lg_te.size(0), -1)
    mu = x_img_tr[tr_idx].mean(0, keepdim=True)
    sd = x_img_tr[tr_idx].std(0, keepdim=True).clamp_min(1e-6)
    x_img_tr = (x_img_tr - mu) / sd
    x_img_te = (x_img_te - mu) / sd

    _, best_img, te_img = train_cls(
        "stack_img", x_img_tr[tr_idx], y_tr[tr_idx], x_img_tr[va_idx], y_tr[va_idx], x_img_te, y_te, args.epochs, 1e-3, device
    )
    results.append({"name": "stack_img", "type": "meta", "filename_required": False, "test_acc": te_img, "val_acc": best_img["acc"], "best_epoch": best_img["epoch"]})

    # multimodal stacker (reference)
    tok_map = fit_token_map(tr_names)
    xt_tr = encode_tokens(tr_names, tok_map)
    xt_te = encode_tokens(te_names, tok_map)
    x_mm_tr = torch.cat([x_img_tr, xt_tr], dim=1)
    x_mm_te = torch.cat([x_img_te, xt_te], dim=1)

    mm_model, best_mm, te_mm = train_cls(
        "stack_mm", x_mm_tr[tr_idx], y_tr[tr_idx], x_mm_tr[va_idx], y_tr[va_idx], x_mm_te, y_te, args.epochs, 1e-3, device
    )
    results.append({"name": "stack_mm", "type": "multimodal", "filename_required": True, "test_acc": te_mm, "val_acc": best_mm["acc"], "best_epoch": best_mm["epoch"]})

    # distill to image-only student
    student = MLP(x_img_tr.size(1), int(y_tr.max().item() + 1)).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(x_img_tr[tr_idx], x_mm_tr[tr_idx], y_tr[tr_idx])
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    best_s = {"acc": 0.0, "state": None, "epoch": 0}
    temp = 2.0
    mm_model.eval()
    for ep in range(1, args.epochs + 1):
        student.train()
        for xb, xmb, yb in dl:
            xb, xmb, yb = xb.to(device), xmb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            student_logits = student(xb)
            with torch.no_grad():
                tout = mm_model(xmb)
            ce = F.cross_entropy(student_logits, yb)
            kd = F.kl_div(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(tout / temp, dim=1),
                reduction="batchmean",
            ) * (temp * temp)
            loss = 0.6 * ce + 0.4 * kd
            loss.backward()
            opt.step()

        student.eval()
        with torch.no_grad():
            va = student(x_img_tr[va_idx].to(device)).cpu()
        va_acc = (va.argmax(1) == y_tr[va_idx]).float().mean().item()
        print(f"[stack_distill] ep {ep:02d} val={va_acc:.4f}")
        if va_acc > best_s["acc"]:
            best_s = {"acc": va_acc, "state": {k: v.detach().cpu() for k, v in student.state_dict().items()}, "epoch": ep}

    student.load_state_dict(best_s["state"])
    student.eval().to(device)
    with torch.no_grad():
        te = student(x_img_te.to(device)).cpu()
    te_s = (te.argmax(1) == y_te).float().mean().item()
    results.append({"name": "stack_img_distill", "type": "meta_distill", "filename_required": False, "test_acc": te_s, "val_acc": best_s["acc"], "best_epoch": best_s["epoch"]})

    # oracle
    oracle = (lg_te.argmax(-1).eq(y_te.unsqueeze(1)).any(1).float().mean().item())

    best_nf = max([r for r in results if not r["filename_required"]], key=lambda x: x["test_acc"])
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_experts": len(names),
        "expert_names": names,
        "oracle": oracle,
        "best_no_filename": best_nf,
        "results": sorted(results, key=lambda x: x["test_acc"], reverse=True),
    }

    (out_dir / "all_expert_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# No Filename (All Experts) Report",
        "",
        f"- Run time: {summary['timestamp']}",
        f"- Experts: {len(names)}",
        "",
        "| Rank | Method | Type | Filename Needed | Test Acc |",
        "|---:|---|---|---|---:|",
    ]
    for i, r in enumerate(summary["results"], 1):
        lines.append(f"| {i} | `{r['name']}` | {r['type']} | {r['filename_required']} | {r['test_acc']:.4f} |")
    lines += [
        "",
        f"- Best no-filename: `{best_nf['name']}` ({best_nf['test_acc']:.4f})",
        f"- Oracle upper bound ({len(names)} experts): {oracle:.4f}",
    ]
    (out_dir / "all_expert_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_dir / "all_expert_summary.json")
    print("saved", out_dir / "all_expert_report.md")


if __name__ == "__main__":
    main()
