import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from mm_automation_common import (
    SimpleTextEncoder,
    build_vocab,
    collect_images_recursive,
    encode_texts,
    fast_visual_cues,
    make_transforms,
    set_seed,
)


class SiteImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform, class_to_idx: Dict[str, int]):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        parent = p.parent.name
        init_label = self.class_to_idx[parent] if parent in self.class_to_idx else -1
        return img, init_label, str(p)


class LabeledImageDataset(Dataset):
    def __init__(self, base: datasets.ImageFolder, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        p, y = self.base.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, y, str(p)


class FewShotAutoLabelModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        vocab_size: int,
        num_classes: int,
        proj_dim: int = 256,
        text_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.image_encoder = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.image_proj = nn.Linear(self.image_encoder.num_features, proj_dim)
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, d_model=text_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.num_classes = num_classes

    def encode_image(self, x):
        z = self.image_encoder(x)
        if z.ndim > 2:
            z = z.flatten(1)
        z = self.image_proj(z)
        return F.normalize(z, dim=1)

    def encode_text(self, text_ids, text_mask):
        z = self.text_encoder(text_ids, text_mask)
        z = self.text_proj(z)
        return F.normalize(z, dim=1)

    def forward(self, images, class_text_ids, class_text_mask):
        im = self.encode_image(images)
        txt = self.encode_text(class_text_ids, class_text_mask)
        logits = im @ txt.t() * self.logit_scale.exp().clamp(max=100.0)
        return logits


@dataclass
class AutoLabelConfig:
    hq_train_dir: str
    site_dir: str
    class_desc_json: str
    out_dir: str
    backbone: str
    img_size: int
    epochs: int
    batch_size: int
    lr: float
    text_max_len: int
    k_shot: int
    alpha_text: float
    conf_threshold: float
    freeze_backbone: bool
    seed: int
    num_workers: int


def collate_site(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    init = torch.tensor([x[1] for x in batch], dtype=torch.long)
    paths = [x[2] for x in batch]
    return imgs, init, paths


def class_texts_from_json(classes: List[str], path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    texts = []
    for c in classes:
        text = str(raw.get(c, "")).strip()
        if not text:
            text = (
                f"Defect class {c}. semiconductor inspection anomaly pattern. "
                "Use texture shape contrast and local artifacts for recognition."
            )
        texts.append(text)
    return texts


def few_shot_indices_by_class(labels: List[int], k: int, seed: int):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[int(y)].append(i)
    picked = {}
    for y, idxs in by.items():
        rng.shuffle(idxs)
        picked[y] = idxs[: min(k, len(idxs))]
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq_train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--site_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--class_desc_json", type=str, default="E:/samsung/Dataset/class_descriptions_template.json")
    parser.add_argument("--out_dir", type=str, default="E:/samsung/Dataset/runs/mm_autolabel")
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b0_ns")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text_max_len", type=int, default=48)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--alpha_text", type=float, default=0.65)
    parser.add_argument("--conf_threshold", type=float, default=0.75)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = AutoLabelConfig(**vars(args))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = out_dir / "pseudo_labeled"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = make_transforms(cfg.img_size, strong_aug=True)

    hq_base = datasets.ImageFolder(cfg.hq_train_dir)
    classes = hq_base.classes
    class_to_idx = hq_base.class_to_idx

    desc_texts = class_texts_from_json(classes, Path(cfg.class_desc_json))
    vocab = build_vocab(desc_texts + [f"class {c}" for c in classes], min_freq=1)
    class_text_ids, class_text_mask = encode_texts(desc_texts, vocab, cfg.text_max_len)
    class_text_ids = class_text_ids.to(device)
    class_text_mask = class_text_mask.to(device)

    train_ds = LabeledImageDataset(hq_base, train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = FewShotAutoLabelModel(
        backbone=cfg.backbone,
        vocab_size=len(vocab),
        num_classes=len(classes),
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Contrastive alignment on HQ labeled data
    history = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images, class_text_ids, class_text_mask)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        row = {
            "epoch": ep,
            "train_loss": total_loss / max(1, total),
            "train_acc": correct / max(1, total),
        }
        history.append(row)
        print(f"[train] epoch={ep} loss={row['train_loss']:.4f} acc={row['train_acc']:.4f}")

    # Build class prototypes: text + few-shot image prototypes
    model.eval()
    with torch.no_grad():
        text_emb = model.encode_text(class_text_ids, class_text_mask)  # [C,D]

    hq_eval_ds = LabeledImageDataset(hq_base, eval_tf)
    hq_eval_loader = DataLoader(hq_eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    img_embs = []
    labels_all = []
    path_all = []
    with torch.no_grad():
        for images, labels, paths in hq_eval_loader:
            images = images.to(device)
            z = model.encode_image(images).cpu()
            img_embs.append(z)
            labels_all.extend(labels.tolist())
            path_all.extend(paths)
    img_embs = torch.cat(img_embs, dim=0)

    few = few_shot_indices_by_class(labels_all, cfg.k_shot, cfg.seed)
    proto_img = []
    for cidx in range(len(classes)):
        idxs = few.get(cidx, [])
        if len(idxs) == 0:
            proto_img.append(torch.zeros_like(img_embs[0]))
        else:
            proto_img.append(img_embs[idxs].mean(dim=0))
    proto_img = F.normalize(torch.stack(proto_img, dim=0).to(device), dim=1)

    alpha = float(cfg.alpha_text)
    prototypes = F.normalize(alpha * text_emb + (1.0 - alpha) * proto_img, dim=1)

    # Site inference
    site_paths = collect_images_recursive(Path(cfg.site_dir))
    site_ds = SiteImageDataset(site_paths, eval_tf, class_to_idx)
    site_loader = DataLoader(
        site_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_site,
    )

    rows = []
    change_counter = Counter()
    kept_counter = Counter()
    confidence_list = []
    with torch.no_grad():
        for images, init_labels, paths in site_loader:
            images = images.to(device)
            z = model.encode_image(images)
            sim = z @ prototypes.t()
            prob = F.softmax(sim, dim=1)
            conf, pred = prob.max(dim=1)

            top3_prob, top3_idx = prob.topk(k=min(3, prob.size(1)), dim=1)
            for i in range(images.size(0)):
                p = Path(paths[i])
                pred_idx = int(pred[i].item())
                pred_cls = classes[pred_idx]
                conf_v = float(conf[i].item())
                confidence_list.append(conf_v)

                init_idx = int(init_labels[i].item())
                init_cls = classes[init_idx] if init_idx >= 0 else ""
                changed = int(init_idx >= 0 and init_idx != pred_idx and conf_v >= cfg.conf_threshold)
                if changed:
                    change_counter[f"{init_cls}->{pred_cls}"] += 1
                else:
                    kept_counter[pred_cls] += 1

                cues = fast_visual_cues(p)
                gen_text = (
                    f"Predicted class {pred_cls}. confidence {conf_v:.4f}. "
                    f"Visual cues: brightness={cues['brightness']}, contrast={cues['contrast']}, "
                    f"edge={cues['edge']}, dark_area={cues['dark_area']}, bright_area={cues['bright_area']}."
                )

                topk_text = "; ".join(
                    [f"{classes[int(top3_idx[i, k])]}:{float(top3_prob[i, k]):.4f}" for k in range(top3_idx.size(1))]
                )

                rows.append(
                    {
                        "image_path": str(p),
                        "initial_label": init_cls,
                        "pred_label": pred_cls,
                        "confidence": conf_v,
                        "changed_by_refinement": changed,
                        "topk": topk_text,
                        "generated_description": gen_text,
                    }
                )

                if conf_v >= cfg.conf_threshold:
                    dst = pseudo_dir / pred_cls / p.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        dst.write_bytes(p.read_bytes())

    # Save outputs
    csv_path = out_dir / "auto_labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "initial_label",
                "pred_label",
                "confidence",
                "changed_by_refinement",
                "topk",
                "generated_description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = out_dir / "auto_descriptions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "config": vars(cfg),
        "num_hq_train": len(hq_base),
        "num_site_images": len(site_ds),
        "num_pseudo_labeled": sum(1 for r in rows if r["confidence"] >= cfg.conf_threshold),
        "mean_confidence": float(sum(confidence_list) / max(1, len(confidence_list))),
        "changed_pairs": dict(change_counter),
        "kept_counter": dict(kept_counter),
        "history": history,
        "files": {
            "auto_labels_csv": str(csv_path),
            "auto_descriptions_jsonl": str(jsonl_path),
            "pseudo_labeled_dir": str(pseudo_dir),
        },
    }
    (out_dir / "autolabel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("saved:", csv_path)
    print("saved:", jsonl_path)
    print("saved:", out_dir / "autolabel_summary.json")


if __name__ == "__main__":
    main()
