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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from mm_automation_common import make_transforms, set_seed
from mm_automation_train_teacher import MultiModalTeacher


class FolderDataset(Dataset):
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


class StudentImageOnly(nn.Module):
    def __init__(self, backbone: str, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        z = self.backbone(x)
        if z.ndim > 2:
            z = z.flatten(1)
        return self.head(z)


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


def evaluate_student(model, loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


def write_report(path: Path, cfg: dict, history: List[dict], best: dict):
    lines = [
        "# Student Distillation Report",
        "",
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Student backbone: {cfg['student_backbone']}",
        f"- Teacher checkpoint: {cfg['teacher_ckpt']}",
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
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--test_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--out_dir", type=str, default="E:/samsung/Dataset/runs/mm_student")
    parser.add_argument("--student_backbone", type=str, default="tf_efficientnet_b0_ns")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    teacher_blob = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher_cfg = teacher_blob["config"]
    classes = teacher_blob["classes"]
    class_text_ids = teacher_blob["class_text_ids"].to(device)
    class_text_mask = teacher_blob["class_text_mask"].to(device)
    vocab_size = len(teacher_blob["vocab"])

    train_tf, eval_tf = make_transforms(args.img_size, strong_aug=True)
    train_ds_full = FolderDataset(Path(args.train_dir), train_tf)
    eval_ds_full = FolderDataset(Path(args.train_dir), eval_tf)
    test_ds = FolderDataset(Path(args.test_dir), eval_tf)

    if train_ds_full.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("Class mapping mismatch between train and test.")

    labels = [y for _, y in train_ds_full.base.samples]
    tr_idx, va_idx = stratified_split(labels, args.val_ratio, args.seed)
    train_ds = Subset(train_ds_full, tr_idx)
    val_ds = Subset(eval_ds_full, va_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    teacher = MultiModalTeacher(
        backbone=teacher_cfg["backbone"],
        vocab_size=vocab_size,
        num_classes=len(classes),
        freeze_backbone=False,
    ).to(device)
    teacher.load_state_dict(teacher_blob["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentImageOnly(
        backbone=args.student_backbone,
        num_classes=len(classes),
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # class weights from train split
    class_counts = [0] * len(classes)
    for i in tr_idx:
        class_counts[labels[i]] += 1
    total = sum(class_counts)
    class_weights = torch.tensor([total / max(1, c) for c in class_counts], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, len(train_loader) * args.warmup_epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best = {"epoch": 0, "val_acc": 0.0, "test_acc": 0.0, "path": str(ckpt_dir / "best_student.pt")}
    history = []
    t = float(args.distill_temp)
    alpha = float(args.distill_alpha)

    for ep in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total_n = 0
        total_c = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                s_logits = student(images)
                with torch.no_grad():
                    t_logits, _, _ = teacher(images, class_text_ids, class_text_mask)
                ce = criterion(s_logits, labels)
                kd = F.kl_div(
                    F.log_softmax(s_logits / t, dim=1),
                    F.softmax(t_logits / t, dim=1),
                    reduction="batchmean",
                ) * (t * t)
                loss = (1.0 - alpha) * ce + alpha * kd
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
            total_c += (s_logits.argmax(1) == labels).sum().item()

        train_loss = total_loss / max(1, total_n)
        train_acc = total_c / max(1, total_n)
        val_acc, val_loss = evaluate_student(student, val_loader, criterion, device)
        test_acc, test_loss = evaluate_student(student, test_loader, criterion, device)
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
                    "model": student.state_dict(),
                    "config": vars(args),
                    "classes": classes,
                    "best": best,
                },
                best["path"],
            )

        write_report(out_dir / "student_report.md", vars(args), history, best)

    final = {
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_test_acc": best["test_acc"],
        "checkpoint": best["path"],
    }
    (out_dir / "student_result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("done:", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
