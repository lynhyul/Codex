import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
    return re.findall(r"[A-Za-z]+|\d+", text.lower())


def build_vocab(texts: List[str], min_freq: int = 1):
    freq: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
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
        toks = tokenize(t)
        seq = [vocab.get(tok, vocab["<unk>"]) for tok in toks][:max_len]
        mask = [1] * len(seq)
        while len(seq) < max_len:
            seq.append(vocab["<pad>"])
            mask.append(0)
        ids.append(seq)
        masks.append(mask)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def make_transforms(img_size: int, strong_aug: bool = False):
    train_ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
    ]
    if strong_aug:
        train_ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    eval_ops = [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def collect_images_recursive(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def fast_visual_cues(image_path: Path):
    img = Image.open(image_path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(128, 128, 3)
    gray = arr.mean(dim=2)
    mean_b = float(gray.mean().item())
    std_b = float(gray.std().item())
    dark_ratio = float((gray < 55).float().mean().item())
    bright_ratio = float((gray > 200).float().mean().item())
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edge_density = float(((gx.abs().mean() + gy.abs().mean()) / 2.0).item() / 64.0)

    def bucket(x, lo, hi):
        if x < lo:
            return "low"
        if x > hi:
            return "high"
        return "moderate"

    return {
        "brightness": bucket(mean_b, 95, 160),
        "contrast": bucket(std_b, 28, 52),
        "dark_area": bucket(dark_ratio, 0.08, 0.28),
        "bright_area": bucket(bright_ratio, 0.05, 0.25),
        "edge": bucket(edge_density, 0.35, 0.7),
    }


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        x = self.emb(ids)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)
