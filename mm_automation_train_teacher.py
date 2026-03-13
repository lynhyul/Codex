import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets

from mm_automation_common import (
    SimpleTextEncoder,
    build_vocab,
    encode_texts,
    make_transforms,
    set_seed,
)


class FolderWithTransform(Dataset):
    def __init__(self, root: Path, transform):
        self.base = datasets.ImageFolder(str(root))
        self.transform = transform
        self.class_to_idx = self.base.class_to_idx

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        p, y = self.base.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, y


def stratified_split(labels: List[int], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(int(y), []).append(i)
    tr, va = [], []
    for _, idxs in by.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        va.extend(idxs[:n_val])
        tr.extend(idxs[n_val:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


class MultiModalTeacher(nn.Module):
    def __init__(
        self,
        backbone: str,
        vocab_size: int,
        num_classes: int,
        proj_dim: int = 384,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.image_encoder = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.image_proj = nn.Linear(self.image_encoder.num_features, proj_dim)
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, d_model=256)
        self.text_proj = nn.Linear(256, proj_dim)
        self.fuse_head = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(proj_dim * 2, num_classes),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.alpha = nn.Parameter(torch.tensor(0.45))

    def encode_image(self, x):
        z = self.image_encoder(x)
        if z.ndim > 2:
            z = z.flatten(1)
        z = self.image_proj(z)
        return F.normalize(z, dim=1)

    def encode_text(self, ids, mask):
        z = self.text_encoder(ids, mask)
        z = self.text_proj(z)
        return F.normalize(z, dim=1)

    def forward(self, images, class_text_ids, class_text_mask):
        im = self.encode_image(images)
        txt = self.encode_text(class_text_ids, class_text_mask)
        sim_logits = im @ txt.t() * self.logit_scale.exp().clamp(max=100.0)
        attn = torch.softmax(sim_logits, dim=1)
        ctx = attn @ txt
        fused = torch.cat([im, ctx, im * ctx, torch.abs(im - ctx)], dim=1)
        fuse_logits = self.fuse_head(fused)
        a = torch.sigmoid(self.alpha)
        logits = a * sim_logits + (1.0 - a) * fuse_logits
        return logits, sim_logits, fuse_logits


def evaluate(model, loader, class_text_ids, class_text_mask, criterion, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, sim_logits, fuse_logits = model(images, class_text_ids, class_text_mask)
            loss = criterion(logits, labels) + 0.3 * criterion(sim_logits, labels) + 0.2 * criterion(
                fuse_logits, labels
            )
            loss_sum += loss.item() * labels.size(0)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


def write_report(path: Path, cfg: dict, history: List[dict], best: dict):
    lines = [
        "# Teacher Training Report",
        "",
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Backbone: {cfg['backbone']}",
        f"- HQ train: {cfg['hq_train_dir']}",
        f"- Pseudo dir: {cfg['pseudo_labeled_dir']}",
        f"- Test dir: {cfg['test_dir']}",
        "",
        "## Best",
        "",
        f"- Best epoch: {best['epoch']}",
        f"- Best val acc: {best['val_acc']:.4f}",
        f"- Best test acc: {best['test_acc']:.4f}",
        "",
        "## Epoch Table",
        "",
        "| Epoch | LR | Train Loss | Train Acc | Val Acc | Val Loss | Test Acc | Test Loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in history:
        lines.append(
            f"| {r['epoch']} | {r['lr']:.6f} | {r['train_loss']:.4f} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['val_loss']:.4f} | {r['test_acc']:.4f} | {r['test_loss']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq_train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--pseudo_labeled_dir", type=str, default="E:/samsung/Dataset/runs/mm_autolabel/pseudo_labeled")
    parser.add_argument("--test_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--class_desc_json", type=str, default="E:/samsung/Dataset/class_descriptions_template.json")
    parser.add_argument("--out_dir", type=str, default="E:/samsung/Dataset/runs/mm_teacher")
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b4_ns")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--text_max_len", type=int, default=48)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = make_transforms(args.img_size, strong_aug=True)

    hq_ds_full = FolderWithTransform(Path(args.hq_train_dir), train_tf)
    hq_eval_full = FolderWithTransform(Path(args.hq_train_dir), eval_tf)
    class_to_idx = hq_ds_full.class_to_idx
    classes = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])

    # class description
    desc = json.loads(Path(args.class_desc_json).read_text(encoding="utf-8"))
    class_texts = [
        str(desc.get(c, f"Defect class {c} semiconductor anomaly pattern with texture and shape cues")).strip()
        for c in classes
    ]
    vocab = build_vocab(class_texts + [f"class {c}" for c in classes], min_freq=1)
    class_text_ids, class_text_mask = encode_texts(class_texts, vocab, args.text_max_len)
    class_text_ids = class_text_ids.to(device)
    class_text_mask = class_text_mask.to(device)

    # split HQ for val
    labels_hq = [y for _, y in hq_ds_full.base.samples]
    tr_idx, va_idx = stratified_split(labels_hq, args.val_ratio, args.seed)
    hq_train = Subset(hq_ds_full, tr_idx)
    hq_val = Subset(hq_eval_full, va_idx)

    # pseudo-labeled data (same class structure)
    pseudo_root = Path(args.pseudo_labeled_dir)
    use_pseudo = pseudo_root.exists() and len(list(pseudo_root.glob("*"))) > 0
    if use_pseudo:
        pseudo_train = FolderWithTransform(pseudo_root, train_tf)
        if pseudo_train.class_to_idx != class_to_idx:
            raise RuntimeError("Class mapping mismatch between HQ and pseudo-labeled data.")
        train_ds = ConcatDataset([hq_train, pseudo_train])
    else:
        train_ds = hq_train

    test_ds = FolderWithTransform(Path(args.test_dir), eval_tf)
    if test_ds.class_to_idx != class_to_idx:
        raise RuntimeError("Class mapping mismatch between train and test.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        hq_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = MultiModalTeacher(
        backbone=args.backbone,
        vocab_size=len(vocab),
        num_classes=len(classes),
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # class weights from HQ train split
    class_counts = [0] * len(classes)
    for i in tr_idx:
        class_counts[labels_hq[i]] += 1
    total = sum(class_counts)
    class_weights = torch.tensor([total / max(1, c) for c in class_counts], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, len(train_loader) * args.warmup_epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best = {"epoch": 0, "val_acc": 0.0, "test_acc": 0.0, "path": str(ckpt_dir / "best_teacher.pt")}
    history = []
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        total_c = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, sim_logits, fuse_logits = model(images, class_text_ids, class_text_mask)
                loss = criterion(logits, labels) + 0.3 * criterion(sim_logits, labels) + 0.2 * criterion(
                    fuse_logits, labels
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
            total_c += (logits.argmax(1) == labels).sum().item()

        train_loss = total_loss / max(1, total_n)
        train_acc = total_c / max(1, total_n)
        val_acc, val_loss = evaluate(model, val_loader, class_text_ids, class_text_mask, criterion, device)
        test_acc, test_loss = evaluate(model, test_loader, class_text_ids, class_text_mask, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        rec = {
            "epoch": ep,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        history.append(rec)
        print(
            f"[epoch {ep:02d}] lr={lr_now:.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"test_acc={test_acc:.4f} best_test={best['test_acc']:.4f}"
        )

        if val_acc >= best["val_acc"]:
            best["epoch"] = ep
            best["val_acc"] = val_acc
            best["test_acc"] = test_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "classes": classes,
                    "vocab": vocab,
                    "class_texts": class_texts,
                    "class_text_ids": class_text_ids.cpu(),
                    "class_text_mask": class_text_mask.cpu(),
                    "best": best,
                },
                best["path"],
            )

        write_report(out_dir / "teacher_report.md", vars(args), history, best)

    final = {
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_test_acc": best["test_acc"],
        "checkpoint": best["path"],
    }
    (out_dir / "teacher_result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("done:", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
