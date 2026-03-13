import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    import timm
except Exception as exc:
    raise RuntimeError("timm is required for this script.") from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_filename_tokens(name: str) -> Tuple[str, str, str, str]:
    m_adi = re.search(r"ADI(\d+)", name)
    m_aoi = re.search(r"AOI(\d+)", name)
    m_vrs = re.search(r"VRS(\d+)", name)
    m_dir = re.search(r"_([A-Z])_AOI", name)
    adi = m_adi.group(1) if m_adi else "NA"
    aoi = m_aoi.group(1) if m_aoi else "NA"
    vrs = m_vrs.group(1) if m_vrs else "NA"
    drc = m_dir.group(1) if m_dir else "NA"
    return adi, aoi, vrs, drc


def stratified_split_indices(labels: List[int], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train_idx, val_idx = [], []
    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def make_transforms(img_size: int, use_strong_aug: bool):
    train_ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if use_strong_aug:
        train_ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    eval_ops = [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def simple_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|\d+", text.lower())


def build_vocab(texts: List[str], min_freq: int = 1):
    freq: Dict[str, int] = {}
    for t in texts:
        for tok in simple_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, cnt in sorted(freq.items()):
        if cnt >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_texts(texts: List[str], vocab: Dict[str, int], max_len: int):
    ids = []
    masks = []
    for t in texts:
        toks = simple_tokenize(t)
        seq = [vocab.get(tok, vocab["<unk>"]) for tok in toks][:max_len]
        mask = [1] * len(seq)
        while len(seq) < max_len:
            seq.append(vocab["<pad>"])
            mask.append(0)
        ids.append(seq)
        masks.append(mask)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


@dataclass
class MetaVocab:
    adi: Dict[str, int]
    aoi: Dict[str, int]
    vrs: Dict[str, int]
    drc: Dict[str, int]

    def cardinalities(self):
        return [len(self.adi), len(self.aoi), len(self.vrs), len(self.drc)]


def build_meta_vocab(names: List[str]):
    vals = [set(), set(), set(), set()]
    for n in names:
        toks = parse_filename_tokens(n)
        for i, v in enumerate(toks):
            vals[i].add(v)

    def to_map(s):
        out = {"<unk>": 0}
        for k in sorted(s):
            out[k] = len(out)
        return out

    return MetaVocab(
        adi=to_map(vals[0]),
        aoi=to_map(vals[1]),
        vrs=to_map(vals[2]),
        drc=to_map(vals[3]),
    )


def encode_meta(name: str, mv: MetaVocab):
    a, b, c, d = parse_filename_tokens(name)
    return torch.tensor(
        [
            mv.adi.get(a, 0),
            mv.aoi.get(b, 0),
            mv.vrs.get(c, 0),
            mv.drc.get(d, 0),
        ],
        dtype=torch.long,
    )


class MultiModalImageFolder(Dataset):
    def __init__(
        self,
        root: Path,
        transform,
        meta_vocab: Optional[MetaVocab],
        use_filename_meta: bool,
    ):
        self.base = datasets.ImageFolder(str(root), transform=transform)
        self.use_filename_meta = use_filename_meta
        self.meta_vocab = meta_vocab
        self.names = [Path(p).name for p, _ in self.base.samples]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        if self.use_filename_meta and self.meta_vocab is not None:
            meta = encode_meta(self.names[idx], self.meta_vocab)
        else:
            meta = torch.zeros(4, dtype=torch.long)
        return img, torch.tensor(y, dtype=torch.long), meta


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        x = self.token_emb(ids) + self.pos_emb[:, : ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return pooled


class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        vocab_size: int,
        text_max_len: int,
        meta_cardinalities: Optional[List[int]],
        pretrained_backbone: bool = True,
        proj_dim: int = 384,
        meta_dim: int = 24,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_meta = meta_cardinalities is not None

        self.image_encoder = timm.create_model(
            backbone, pretrained=pretrained_backbone, num_classes=0, global_pool="avg"
        )
        img_dim = self.image_encoder.num_features
        self.image_proj = nn.Linear(img_dim, proj_dim)

        self.text_encoder = TextEncoder(vocab_size=vocab_size, max_len=text_max_len, d_model=256)
        self.text_proj = nn.Linear(256, proj_dim)

        if self.use_meta:
            self.meta_embs = nn.ModuleList([nn.Embedding(c, meta_dim) for c in meta_cardinalities])
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim * 4, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )
            fusion_dim = proj_dim * 5
        else:
            fusion_dim = proj_dim * 4

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, num_classes),
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))
        self.combine_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def encode_meta(self, meta_ids: torch.Tensor):
        emb = [self.meta_embs[i](meta_ids[:, i]) for i in range(4)]
        z = torch.cat(emb, dim=1)
        return self.meta_proj(z)

    def forward(
        self,
        images: torch.Tensor,
        class_text_ids: torch.Tensor,
        class_text_mask: torch.Tensor,
        meta_ids: Optional[torch.Tensor] = None,
    ):
        img = self.image_encoder(images)
        if img.ndim > 2:
            img = img.flatten(1)
        img = self.image_proj(img)

        txt = self.text_encoder(class_text_ids, class_text_mask)
        txt = self.text_proj(txt)

        img_n = F.normalize(img, dim=1)
        txt_n = F.normalize(txt, dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        sim_logits = img_n @ txt_n.t() * scale

        attn = torch.softmax(sim_logits, dim=1)
        ctx = attn @ txt

        pieces = [img, ctx, img * ctx, torch.abs(img - ctx)]
        if self.use_meta and meta_ids is not None:
            meta_z = self.encode_meta(meta_ids)
            pieces.append(meta_z)
        fused = torch.cat(pieces, dim=1)
        fuse_logits = self.fusion_head(fused)

        alpha = torch.sigmoid(self.combine_alpha)
        logits = alpha * sim_logits + (1.0 - alpha) * fuse_logits
        return logits, sim_logits, fuse_logits


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_text_ids: torch.Tensor,
    class_text_mask: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    tta_flip: bool = False,
    max_batches: int = 0,
):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for bidx, (images, labels, meta) in enumerate(loader):
            if max_batches > 0 and bidx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            logits, sim_logits, fuse_logits = model(images, class_text_ids, class_text_mask, meta)
            if tta_flip:
                flipped = torch.flip(images, dims=[3])
                lg2, s2, f2 = model(flipped, class_text_ids, class_text_mask, meta)
                logits = (logits + lg2) / 2.0
                sim_logits = (sim_logits + s2) / 2.0
                fuse_logits = (fuse_logits + f2) / 2.0

            loss = criterion(logits, labels) + 0.35 * criterion(sim_logits, labels) + 0.25 * criterion(
                fuse_logits, labels
            )
            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / max(1, total), total_loss / max(1, total)


def write_report(path: Path, config: dict, history: List[dict], best: dict):
    lines = []
    lines.append("# Multimodal Training Report")
    lines.append("")
    lines.append(f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Train dir: {config['train_dir']}")
    lines.append(f"- Test dir: {config['test_dir']}")
    lines.append(f"- Backbone: {config['backbone']}")
    lines.append(f"- Use filename meta: {config.get('use_filename_meta', 'unknown')}")
    lines.append("")
    lines.append("## Best")
    lines.append("")
    lines.append(f"- Best epoch: {best['epoch']}")
    lines.append(f"- Best val acc: {best['val_acc']:.4f}")
    lines.append(f"- Best test acc: {best['test_acc']:.4f}")
    lines.append("")
    lines.append("## Epoch Metrics")
    lines.append("")
    lines.append("| Epoch | LR | Train Loss | Train Acc | Val Acc | Val Loss | Test Acc | Test Loss |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in history:
        lines.append(
            f"| {r['epoch']} | {r['lr']:.6f} | {r['train_loss']:.4f} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['val_loss']:.4f} | {r['test_acc']:.4f} | {r['test_loss']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--test_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--desc_json", type=str, default="E:/samsung/Dataset/class_descriptions_template.json")
    parser.add_argument("--out_dir", type=str, default="E:/samsung/Dataset/runs/multimodal_desc")

    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b4_ns")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta_flip", action="store_true")
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--no_filename_meta", action="store_true")
    parser.add_argument("--text_max_len", type=int, default=40)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = out_dir / "history.jsonl"
    report_md = out_dir / "performance_report.md"

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    desc_path = Path(args.desc_json)

    train_tf, eval_tf = make_transforms(args.img_size, use_strong_aug=True)

    # Base datasets only for class and splitting metadata
    base_train = datasets.ImageFolder(str(train_dir))
    classes = base_train.classes
    labels = base_train.targets
    train_idx, val_idx = stratified_split_indices(labels, args.val_ratio, args.seed)

    # Class descriptions
    desc_json = json.loads(desc_path.read_text(encoding="utf-8"))
    desc_texts = []
    for c in classes:
        if c in desc_json and str(desc_json[c]).strip():
            desc_texts.append(str(desc_json[c]).strip())
        else:
            desc_texts.append(f"class {c} defect pattern in semiconductor inspection image")

    vocab = build_vocab(desc_texts + [f"class {c}" for c in classes], min_freq=1)
    class_text_ids, class_text_mask = encode_texts(desc_texts, vocab, args.text_max_len)

    # Filename meta vocabulary from train set
    use_filename_meta = not args.no_filename_meta
    meta_vocab = None
    if use_filename_meta:
        train_names = [Path(p).name for p, _ in base_train.samples]
        meta_vocab = build_meta_vocab(train_names)

    train_ds_full = MultiModalImageFolder(train_dir, train_tf, meta_vocab, use_filename_meta)
    val_ds_full = MultiModalImageFolder(train_dir, eval_tf, meta_vocab, use_filename_meta)
    test_ds = MultiModalImageFolder(test_dir, eval_tf, meta_vocab, use_filename_meta)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Class weights from train split
    class_counts = [0] * len(classes)
    for i in train_idx:
        class_counts[labels[i]] += 1
    total_cnt = sum(class_counts)
    class_weights = torch.tensor(
        [total_cnt / max(1, c) for c in class_counts], dtype=torch.float32, device=device
    )

    model = MultiModalClassifier(
        backbone=args.backbone,
        num_classes=len(classes),
        vocab_size=len(vocab),
        text_max_len=args.text_max_len,
        meta_cardinalities=meta_vocab.cardinalities() if use_filename_meta else None,
        pretrained_backbone=not args.no_pretrained_backbone,
        proj_dim=384,
        meta_dim=24,
        dropout=0.25,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, len(train_loader) * args.warmup_epochs)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    class_text_ids = class_text_ids.to(device)
    class_text_mask = class_text_mask.to(device)

    config_to_save = vars(args).copy()
    config_to_save["use_filename_meta"] = use_filename_meta
    config_to_save["classes"] = classes
    config_to_save["num_train"] = len(train_idx)
    config_to_save["num_val"] = len(val_idx)
    config_to_save["num_test"] = len(test_ds)
    (out_dir / "config.json").write_text(json.dumps(config_to_save, indent=2), encoding="utf-8")

    best = {
        "epoch": 0,
        "val_acc": 0.0,
        "test_acc": 0.0,
        "path": str(ckpt_dir / "best_multimodal.pt"),
    }
    history = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step_idx, (images, labels_batch, meta) in enumerate(train_loader):
            if args.max_train_steps > 0 and step_idx >= args.max_train_steps:
                break
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, sim_logits, fuse_logits = model(images, class_text_ids, class_text_mask, meta)
                loss = criterion(logits, labels_batch)
                loss = loss + 0.35 * criterion(sim_logits, labels_batch) + 0.25 * criterion(
                    fuse_logits, labels_batch
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += loss.item() * labels_batch.size(0)
            pred = logits.argmax(dim=1)
            running_correct += (pred == labels_batch).sum().item()
            running_total += labels_batch.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_acc, val_loss = evaluate(
                model,
                val_loader,
                class_text_ids,
                class_text_mask,
                criterion,
                device,
                tta_flip=args.tta_flip,
                max_batches=args.max_eval_batches,
            )
            test_acc, test_loss = evaluate(
                model,
                test_loader,
                class_text_ids,
                class_text_mask,
                criterion,
                device,
                tta_flip=args.tta_flip,
                max_batches=args.max_eval_batches,
            )
        else:
            val_acc, val_loss = 0.0, 0.0
            test_acc, test_loss = 0.0, 0.0

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        history.append(row)
        with history_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if val_acc >= best["val_acc"]:
            best["epoch"] = epoch
            best["val_acc"] = val_acc
            best["test_acc"] = test_acc
            ckpt = {
                "model": model.state_dict(),
                "config": config_to_save,
                "vocab": vocab,
                "class_text_ids": class_text_ids.cpu(),
                "class_text_mask": class_text_mask.cpu(),
                "meta_vocab": None
                if meta_vocab is None
                else {
                    "adi": meta_vocab.adi,
                    "aoi": meta_vocab.aoi,
                    "vrs": meta_vocab.vrs,
                    "drc": meta_vocab.drc,
                },
                "best": best,
            }
            torch.save(ckpt, best["path"])

        write_report(report_md, config_to_save, history, best)
        print(
            f"[Epoch {epoch:02d}] lr={lr_now:.6f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
            f"best_test={best['test_acc']:.4f}"
        )

    final = {
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_test_acc": best["test_acc"],
        "checkpoint": best["path"],
        "report": str(report_md),
    }
    (out_dir / "final_result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("done:", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
