import argparse
import json
import math
import random
import time
from dataclasses import dataclass
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


def stratified_split(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    class_to_idx: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        class_to_idx.setdefault(int(y), []).append(i)
    tr, va = [], []
    for _, idxs in class_to_idx.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        va.extend(idxs[:n_val])
        tr.extend(idxs[n_val:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def make_tf(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(model_name: str, num_classes: int) -> Optional[nn.Module]:
    if model_name.startswith("timm:"):
        import timm

        return timm.create_model(model_name.split(":", 1)[1], pretrained=False, num_classes=num_classes)
    if model_name == "resnet18":
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "resnet34":
        m = models.resnet34(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "resnet50":
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "efficientnet_b0" and hasattr(models, "efficientnet_b0"):
        m = models.efficientnet_b0(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if model_name == "efficientnet_b3" and hasattr(models, "efficientnet_b3"):
        m = models.efficientnet_b3(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    return None


def parse_tokens(name: str) -> Tuple[str, str, str, str]:
    import re

    m_adi = re.search(r"ADI(\d+)", name)
    m_aoi = re.search(r"AOI(\d+)", name)
    m_vrs = re.search(r"VRS(\d+)", name)
    m_dir = re.search(r"_([A-Z])_AOI", name)
    return (
        m_adi.group(1) if m_adi else "NA",
        m_aoi.group(1) if m_aoi else "NA",
        m_vrs.group(1) if m_vrs else "NA",
        m_dir.group(1) if m_dir else "NA",
    )


def build_token_encoder(train_names: List[str]):
    values = [set(), set(), set(), set()]
    for n in train_names:
        toks = parse_tokens(n)
        for i in range(4):
            values[i].add(toks[i])
    maps = []
    for i in range(4):
        v = sorted(values[i])
        maps.append({k: j for j, k in enumerate(v)})
    return maps


def encode_tokens(names: List[str], maps) -> torch.Tensor:
    dims = [len(m) + 1 for m in maps]
    total = sum(dims)
    out = torch.zeros((len(names), total), dtype=torch.float32)
    offsets = [0]
    for d in dims[:-1]:
        offsets.append(offsets[-1] + d)
    for i, n in enumerate(names):
        toks = parse_tokens(n)
        for j in range(4):
            mp = maps[j]
            idx = mp.get(toks[j], len(mp))
            out[i, offsets[j] + idx] = 1.0
    return out


@dataclass
class ExpertPred:
    name: str
    checkpoint: Path
    model_name: str
    img_size: int
    tta: bool
    val_acc: float
    logits_train: torch.Tensor
    logits_test: torch.Tensor


def infer_checkpoint(
    ckpt_path: Path,
    train_dir: Path,
    test_dir: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Optional[ExpertPred]:
    cache_train = cache_dir / f"train_{ckpt_path.stem}.pt"
    cache_test = cache_dir / f"test_{ckpt_path.stem}.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name")
    if not model_name:
        return None
    model_probe = build_model(model_name, 26)
    if model_probe is None:
        return None

    img_size = int(cfg.get("img_size", 224))
    tta = bool(cfg.get("tta", False))

    def run_split(data_dir: Path, split_cache: Path):
        if split_cache.exists():
            blob = torch.load(split_cache, map_location="cpu")
            return blob["logits"], blob["labels"], blob["names"], blob["classes"]

        ds = datasets.ImageFolder(str(data_dir), transform=make_tf(img_size))
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        model = build_model(model_name, len(ds.classes))
        if model is None:
            raise RuntimeError(f"Unsupported model: {model_name}")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        logits_all = []
        labels_all = []
        names = [Path(p).name for p, _ in ds.samples]
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                if tta:
                    logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
                logits_all.append(logits.cpu())
                labels_all.append(y.cpu())
        logits = torch.cat(logits_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        torch.save(
            {
                "logits": logits,
                "labels": labels,
                "names": names,
                "classes": ds.classes,
            },
            split_cache,
        )
        return logits, labels, names, ds.classes

    lg_tr, y_tr, names_tr, classes_tr = run_split(train_dir, cache_train)
    lg_te, y_te, names_te, classes_te = run_split(test_dir, cache_test)

    if classes_tr != classes_te:
        raise RuntimeError("Class mismatch between train and test")

    return ExpertPred(
        name=ckpt_path.stem,
        checkpoint=ckpt_path,
        model_name=model_name,
        img_size=img_size,
        tta=tta,
        val_acc=float(ckpt.get("val_acc", 0.0)),
        logits_train=lg_tr,
        logits_test=lg_te,
    )


def build_img_features(logits_stack: torch.Tensor) -> torch.Tensor:
    # logits_stack: [N, M, C]
    probs = torch.softmax(logits_stack, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    conf = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    flat_logits = logits_stack.reshape(logits_stack.size(0), -1)
    feat = torch.cat([flat_logits, conf, margin, entropy], dim=1)
    return feat


class GateNet(nn.Module):
    def __init__(self, in_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_experts),
        )

    def forward(self, x):
        return self.net(x)


def combined_logits(gate_logits: torch.Tensor, expert_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # gate_logits: [B, M], expert_logits: [B, M, C]
    w = torch.softmax(gate_logits, dim=1)
    out = (w.unsqueeze(-1) * expert_logits).sum(dim=1)
    return out, w


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    p = logits.argmax(dim=1)
    return (p == y).float().mean().item()


def fit_gate(
    name: str,
    x_tr: torch.Tensor,
    x_va: torch.Tensor,
    exp_tr: torch.Tensor,
    exp_va: torch.Tensor,
    y_tr: torch.Tensor,
    y_va: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    teacher: Optional[nn.Module] = None,
    teacher_x_tr: Optional[torch.Tensor] = None,
    teacher_x_va: Optional[torch.Tensor] = None,
    alpha_ce: float = 0.7,
    alpha_kl: float = 0.3,
    temp: float = 2.0,
):
    model = GateNet(x_tr.size(1), exp_tr.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_ds = TensorDataset(x_tr, exp_tr, y_tr)
    va_ds = TensorDataset(x_va, exp_va, y_va)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    best = {
        "val_acc": 0.0,
        "state": None,
        "epoch": 0,
    }

    teacher.eval() if teacher is not None else None

    for ep in range(1, epochs + 1):
        model.train()
        for bi, (xb, eb, yb) in enumerate(tr_loader):
            xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            gl = model(xb)
            out, _ = combined_logits(gl, eb)
            loss = F.cross_entropy(out, yb)

            if teacher is not None and teacher_x_tr is not None:
                start = bi * batch_size
                end = start + xb.size(0)
                tx = teacher_x_tr[start:end].to(device)
                with torch.no_grad():
                    t_gl = teacher(tx)
                    t_out, _ = combined_logits(t_gl, eb)
                kd = F.kl_div(
                    F.log_softmax(out / temp, dim=1),
                    F.softmax(t_out / temp, dim=1),
                    reduction="batchmean",
                ) * (temp * temp)
                loss = alpha_ce * loss + alpha_kl * kd

            loss.backward()
            opt.step()

        model.eval()
        logits_va = []
        ys_va = []
        with torch.no_grad():
            for xb, eb, yb in va_loader:
                xb, eb = xb.to(device), eb.to(device)
                gl = model(xb)
                out, _ = combined_logits(gl, eb)
                logits_va.append(out.cpu())
                ys_va.append(yb)
        logits_va = torch.cat(logits_va)
        ys_va = torch.cat(ys_va)
        va_acc = accuracy(logits_va, ys_va)
        print(f"[{name}] epoch {ep:02d} val_acc={va_acc:.4f}")

        if va_acc > best["val_acc"]:
            best["val_acc"] = va_acc
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best["epoch"] = ep

    model.load_state_dict(best["state"])
    model.to(device)
    model.eval()
    return model, best


def eval_gate(model: nn.Module, x: torch.Tensor, exp: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    ds = TensorDataset(x, exp, y)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    outs = []
    ys = []
    with torch.no_grad():
        for xb, eb, yb in loader:
            xb, eb = xb.to(device), eb.to(device)
            gl = model(xb)
            out, _ = combined_logits(gl, eb)
            outs.append(out.cpu())
            ys.append(yb)
    out = torch.cat(outs)
    yy = torch.cat(ys)
    return accuracy(out, yy)


def weighted_average_logits(logits_stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # logits_stack [N,M,C], weights [M]
    w = torch.softmax(weights, dim=0)
    return (logits_stack * w.view(1, -1, 1)).sum(dim=1)


def fit_scalar_weights(logits_tr: torch.Tensor, y_tr: torch.Tensor, logits_va: torch.Tensor, y_va: torch.Tensor):
    m = logits_tr.size(1)
    w = nn.Parameter(torch.zeros(m))
    opt = torch.optim.Adam([w], lr=0.05)
    best = None
    best_acc = 0.0
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        out = weighted_average_logits(logits_tr, w)
        loss = F.cross_entropy(out, y_tr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            va_out = weighted_average_logits(logits_va, w)
            va_acc = accuracy(va_out, y_va)
            if va_acc > best_acc:
                best_acc = va_acc
                best = w.detach().clone()
    return best, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--test_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--runs_dir", type=str, default="E:/samsung/Dataset/runs")
    parser.add_argument("--out_dir", type=str, default="E:/samsung/Dataset/runs/no_filename_eval")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Expert list: best diversity/accuracy mix
    expert_ckpts = [
        runs_dir / "best_t9_timm_b4_320.pt",
        runs_dir / "best_t12_swin_tiny.pt",
        runs_dir / "best_t5_effb3.pt",
        runs_dir / "best_t11_convnext_small.pt",
        runs_dir / "best_t4_effb0.pt",
    ]
    expert_ckpts = [p for p in expert_ckpts if p.exists() and p.stat().st_size > 0]

    experts: List[ExpertPred] = []
    train_names_ref = None
    test_names_ref = None
    y_train_ref = None
    y_test_ref = None

    for cp in expert_ckpts:
        print(f"[infer] {cp.name}")
        ex = infer_checkpoint(
            cp,
            train_dir=train_dir,
            test_dir=test_dir,
            cache_dir=cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if ex is None:
            print(f"skip {cp.name}")
            continue

        blob_tr = torch.load(cache_dir / f"train_{cp.stem}.pt", map_location="cpu")
        blob_te = torch.load(cache_dir / f"test_{cp.stem}.pt", map_location="cpu")

        if train_names_ref is None:
            train_names_ref = blob_tr["names"]
            test_names_ref = blob_te["names"]
            y_train_ref = blob_tr["labels"].long()
            y_test_ref = blob_te["labels"].long()
        else:
            if train_names_ref != blob_tr["names"]:
                raise RuntimeError("train sample order mismatch")
            if test_names_ref != blob_te["names"]:
                raise RuntimeError("test sample order mismatch")
            if not torch.equal(y_train_ref, blob_tr["labels"].long()):
                raise RuntimeError("train label mismatch")
            if not torch.equal(y_test_ref, blob_te["labels"].long()):
                raise RuntimeError("test label mismatch")

        experts.append(ex)

    if not experts:
        raise SystemExit("No experts loaded")

    y_train = y_train_ref
    y_test = y_test_ref
    train_names = train_names_ref
    test_names = test_names_ref

    # Stack logits
    logits_train = torch.stack([e.logits_train for e in experts], dim=1)  # [N,M,C]
    logits_test = torch.stack([e.logits_test for e in experts], dim=1)

    # Split train/val
    tr_idx, va_idx = stratified_split(y_train.tolist(), args.val_ratio, args.seed)
    tr_idx = torch.tensor(tr_idx, dtype=torch.long)
    va_idx = torch.tensor(va_idx, dtype=torch.long)

    # Features
    x_img_train = build_img_features(logits_train)
    x_img_test = build_img_features(logits_test)

    tok_maps = build_token_encoder(train_names)
    x_tok_train = encode_tokens(train_names, tok_maps)
    x_tok_test = encode_tokens(test_names, tok_maps)

    # Standardize image features using train stats
    mu = x_img_train[tr_idx].mean(dim=0, keepdim=True)
    std = x_img_train[tr_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    x_img_train = (x_img_train - mu) / std
    x_img_test = (x_img_test - mu) / std

    # Baseline metrics per expert
    results = []
    for i, ex in enumerate(experts):
        acc_tr = accuracy(logits_train[:, i, :], y_train)
        acc_te = accuracy(logits_test[:, i, :], y_test)
        results.append({
            "name": ex.name,
            "type": "expert",
            "model": ex.model_name,
            "val_acc_ckpt": ex.val_acc,
            "train_acc": acc_tr,
            "test_acc": acc_te,
            "filename_required": False,
        })
        print(f"[expert] {ex.name}: test_acc={acc_te:.4f}")

    # Best expert baseline
    best_expert = max(results, key=lambda r: r["test_acc"])

    # Uniform average
    uni_logits_test = logits_test.mean(dim=1)
    uni_test_acc = accuracy(uni_logits_test, y_test)
    uni_logits_tr = logits_train[tr_idx].mean(dim=1)
    uni_val_acc = accuracy(logits_train[va_idx].mean(dim=1), y_train[va_idx])
    results.append({
        "name": "uniform_avg",
        "type": "ensemble",
        "model": "+".join([e.name for e in experts]),
        "val_acc_ckpt": None,
        "train_acc": accuracy(uni_logits_tr, y_train[tr_idx]),
        "test_acc": uni_test_acc,
        "filename_required": False,
        "val_acc_meta": uni_val_acc,
    })
    print(f"[ensemble] uniform avg test_acc={uni_test_acc:.4f}")

    # Learned scalar weights
    w_best, va_acc_w = fit_scalar_weights(
        logits_train[tr_idx], y_train[tr_idx], logits_train[va_idx], y_train[va_idx]
    )
    w_test_logits = weighted_average_logits(logits_test, w_best)
    w_test_acc = accuracy(w_test_logits, y_test)
    results.append({
        "name": "scalar_weighted_avg",
        "type": "ensemble",
        "model": "+".join([e.name for e in experts]),
        "val_acc_ckpt": None,
        "train_acc": accuracy(weighted_average_logits(logits_train[tr_idx], w_best), y_train[tr_idx]),
        "test_acc": w_test_acc,
        "filename_required": False,
        "val_acc_meta": va_acc_w,
        "weights": torch.softmax(w_best, dim=0).tolist(),
    })
    print(f"[ensemble] scalar weighted test_acc={w_test_acc:.4f}")

    # Image-only gate (no filename)
    gate_img, best_img = fit_gate(
        name="img_gate",
        x_tr=x_img_train[tr_idx],
        x_va=x_img_train[va_idx],
        exp_tr=logits_train[tr_idx],
        exp_va=logits_train[va_idx],
        y_tr=y_train[tr_idx],
        y_va=y_train[va_idx],
        epochs=args.epochs,
        lr=1e-3,
        batch_size=256,
        device=device,
    )
    img_gate_test_acc = eval_gate(gate_img, x_img_test, logits_test, y_test, device)
    results.append({
        "name": "img_gate",
        "type": "meta",
        "model": "gate(image_features)->weighted_experts",
        "val_acc_ckpt": None,
        "train_acc": eval_gate(gate_img, x_img_train[tr_idx], logits_train[tr_idx], y_train[tr_idx], device),
        "test_acc": img_gate_test_acc,
        "filename_required": False,
        "val_acc_meta": best_img["val_acc"],
        "best_epoch": best_img["epoch"],
    })
    print(f"[meta] img gate test_acc={img_gate_test_acc:.4f}")

    # Multimodal teacher (image + token)
    x_mm_train = torch.cat([x_img_train, x_tok_train], dim=1)
    x_mm_test = torch.cat([x_img_test, x_tok_test], dim=1)

    gate_mm, best_mm = fit_gate(
        name="mm_teacher",
        x_tr=x_mm_train[tr_idx],
        x_va=x_mm_train[va_idx],
        exp_tr=logits_train[tr_idx],
        exp_va=logits_train[va_idx],
        y_tr=y_train[tr_idx],
        y_va=y_train[va_idx],
        epochs=args.epochs,
        lr=1e-3,
        batch_size=256,
        device=device,
    )
    mm_test_acc = eval_gate(gate_mm, x_mm_test, logits_test, y_test, device)
    results.append({
        "name": "mm_teacher",
        "type": "multimodal",
        "model": "gate(image_features+tokens)->weighted_experts",
        "val_acc_ckpt": None,
        "train_acc": eval_gate(gate_mm, x_mm_train[tr_idx], logits_train[tr_idx], y_train[tr_idx], device),
        "test_acc": mm_test_acc,
        "filename_required": True,
        "val_acc_meta": best_mm["val_acc"],
        "best_epoch": best_mm["epoch"],
    })
    print(f"[multimodal] teacher test_acc={mm_test_acc:.4f}")

    # Distilled student (no filename): train with CE+KD from multimodal teacher
    gate_distill, best_distill = fit_gate(
        name="img_gate_distill",
        x_tr=x_img_train[tr_idx],
        x_va=x_img_train[va_idx],
        exp_tr=logits_train[tr_idx],
        exp_va=logits_train[va_idx],
        y_tr=y_train[tr_idx],
        y_va=y_train[va_idx],
        epochs=args.epochs,
        lr=1e-3,
        batch_size=256,
        device=device,
        teacher=gate_mm,
        teacher_x_tr=x_mm_train[tr_idx],
        teacher_x_va=x_mm_train[va_idx],
        alpha_ce=0.6,
        alpha_kl=0.4,
        temp=2.0,
    )
    distill_test_acc = eval_gate(gate_distill, x_img_test, logits_test, y_test, device)
    results.append({
        "name": "img_gate_distill",
        "type": "meta_distill",
        "model": "image-only gate distilled from multimodal teacher",
        "val_acc_ckpt": None,
        "train_acc": eval_gate(gate_distill, x_img_train[tr_idx], logits_train[tr_idx], y_train[tr_idx], device),
        "test_acc": distill_test_acc,
        "filename_required": False,
        "val_acc_meta": best_distill["val_acc"],
        "best_epoch": best_distill["epoch"],
    })
    print(f"[distill] image-only student test_acc={distill_test_acc:.4f}")

    # Oracle upper bound over selected experts (filename-free theoretical upper bound for routing)
    pred_experts_test = logits_test.argmax(dim=-1)  # [N,M]
    oracle = (pred_experts_test.eq(y_test.unsqueeze(1)).any(dim=1).float().mean().item())

    # Save summary
    best_no_filename = max([r for r in results if not r["filename_required"]], key=lambda r: r["test_acc"])
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "num_experts": len(experts),
        "experts": [
            {
                "name": e.name,
                "checkpoint": str(e.checkpoint),
                "model": e.model_name,
                "img_size": e.img_size,
                "tta": e.tta,
                "val_acc": e.val_acc,
            }
            for e in experts
        ],
        "oracle_selected_experts": oracle,
        "best_no_filename": best_no_filename,
        "results": sorted(results, key=lambda r: r["test_acc"], reverse=True),
        "assumption": "Test filename unavailable for deployment. filename_required=True methods are reference only.",
    }

    (out_dir / "no_filename_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown report
    lines = []
    lines.append("# No-Filename Evaluation Report")
    lines.append("")
    lines.append(f"- Run time: {summary['timestamp']}")
    lines.append(f"- Device: {summary['device']}")
    lines.append("- Assumption: test filename is NOT available at inference")
    lines.append("")
    lines.append("## Performance Table")
    lines.append("")
    lines.append("| Rank | Method | Type | Filename Needed | Test Acc |")
    lines.append("|---:|---|---|---|---:|")
    for i, r in enumerate(summary["results"], start=1):
        lines.append(
            f"| {i} | `{r['name']}` | {r['type']} | {r['filename_required']} | {r['test_acc']:.4f} |"
        )
    lines.append("")
    lines.append("## Key Points")
    lines.append("")
    lines.append(f"- Best no-filename method: `{best_no_filename['name']}` ({best_no_filename['test_acc']:.4f})")
    lines.append(f"- Oracle upper bound with selected experts: {oracle:.4f}")
    lines.append("- `mm_teacher` is multimodal reference (image+token), not deployable when filename is absent.")
    lines.append("- `img_gate_distill` is deployable image-only student distilled from multimodal teacher.")
    lines.append("")
    (out_dir / "no_filename_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved:", out_dir / "no_filename_summary.json")
    print("saved:", out_dir / "no_filename_report.md")


if __name__ == "__main__":
    main()
