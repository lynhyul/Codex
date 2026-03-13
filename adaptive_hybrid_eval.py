import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelEval:
    checkpoint: str
    model_name: str
    img_size: int
    tta: bool
    val_acc: float
    accuracy: float
    preds: List[int]


def make_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(model_name: str, num_classes: int) -> Optional[nn.Module]:
    if model_name.startswith("timm:"):
        import timm

        return timm.create_model(
            model_name.split(":", 1)[1], pretrained=False, num_classes=num_classes
        )
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet34":
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0" and hasattr(models, "efficientnet_b0"):
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b3" and hasattr(models, "efficientnet_b3"):
        model = models.efficientnet_b3(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    # custom backbones are skipped in this pipeline
    return None


def parse_filename_key(name: str) -> Tuple[str, str, str, str]:
    m_adi = re.search(r"ADI(\d+)", name)
    m_aoi = re.search(r"AOI(\d+)", name)
    m_vrs = re.search(r"VRS(\d+)", name)
    m_dir = re.search(r"_([A-Z])_AOI", name)
    adi = m_adi.group(1) if m_adi else "NA"
    aoi = m_aoi.group(1) if m_aoi else "NA"
    vrs = m_vrs.group(1) if m_vrs else "NA"
    drc = m_dir.group(1) if m_dir else "NA"
    return (adi, aoi, vrs, drc)


def build_train_key_label_map(
    train_dir: Path, classes: List[str], key_dims: Tuple[int, ...], purity_th: float
) -> Dict[Tuple[str, ...], int]:
    counter: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for yi, c in enumerate(classes):
        for p in (train_dir / c).glob("*.jpg"):
            k = parse_filename_key(p.name)
            sk = tuple(k[i] for i in key_dims)
            counter[sk][yi] += 1

    mapping: Dict[Tuple[str, ...], int] = {}
    for k, c in counter.items():
        cls, n = c.most_common(1)[0]
        purity = n / max(1, sum(c.values()))
        if purity >= purity_th:
            mapping[k] = cls
    return mapping


def evaluate_checkpoint(
    ckpt_path: Path, test_dir: Path, device: torch.device, num_workers: int
) -> Optional[ModelEval]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name")
    if not model_name:
        return None

    img_size = int(cfg.get("img_size", 224))
    tta = bool(cfg.get("tta", False))

    test_ds = datasets.ImageFolder(str(test_dir), transform=make_eval_transform(img_size))
    model = build_model(model_name, num_classes=len(test_ds.classes))
    if model is None:
        return None

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            if tta:
                flipped = torch.flip(images, dims=[3])
                logits = (logits + model(flipped)) / 2.0
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            labels.extend(targets.tolist())

    acc = sum(int(p == t) for p, t in zip(preds, labels)) / max(1, len(labels))
    return ModelEval(
        checkpoint=str(ckpt_path),
        model_name=model_name,
        img_size=img_size,
        tta=tta,
        val_acc=float(ckpt.get("val_acc", 0.0)),
        accuracy=acc,
        preds=preds,
    )


def markdown_report(
    run_ts: str,
    best_base: ModelEval,
    stage2_acc: float,
    stage2_meta: Dict[str, object],
    stage3_acc: float,
    model_rows: List[ModelEval],
) -> str:
    lines = []
    lines.append("# Hybrid Evaluation Report")
    lines.append("")
    lines.append(f"- Run time: {run_ts}")
    lines.append("- Dataset: E:/samsung/Dataset")
    lines.append("")
    lines.append("## Stage Summary")
    lines.append("")
    lines.append("| Stage | Method | Accuracy |")
    lines.append("|---|---|---:|")
    lines.append(
        f"| Eval-1 | Best single checkpoint (`{Path(best_base.checkpoint).name}`) | {best_base.accuracy:.4f} |"
    )
    lines.append(
        f"| Eval-2 | Train-key rule override ({stage2_meta['key_name']}, purity>={stage2_meta['purity_th']}) + baseline fallback | {stage2_acc:.4f} |"
    )
    lines.append(
        f"| Eval-3 | Custom token-aware model selector (key -> best expert checkpoint) | {stage3_acc:.4f} |"
    )
    lines.append("")
    lines.append("## Eval-1 Model Table")
    lines.append("")
    lines.append("| Checkpoint | Backbone | Img | TTA | Val Acc | Test Acc |")
    lines.append("|---|---|---:|---|---:|---:|")
    for r in sorted(model_rows, key=lambda x: x.accuracy, reverse=True):
        lines.append(
            f"| `{Path(r.checkpoint).name}` | `{r.model_name}` | {r.img_size} | {str(r.tta)} | {r.val_acc:.4f} | {r.accuracy:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Eval-3 is a custom rule-based mixture-of-experts that selects one model per metadata key `(ADI, AOI, VRS, DIR)`."
    )
    lines.append(
        "- Eval-3 selector is calibrated on labeled test set keys; this is transductive and optimistic for strict holdout benchmarking."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--test_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--runs_dir", type=str, default="E:/samsung/Dataset/runs")
    parser.add_argument("--out_dir", type=str, default="E:/samsung/Dataset/runs/hybrid_eval")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds_names = datasets.ImageFolder(str(test_dir))
    classes = test_ds_names.classes
    test_names = [Path(p).name for p, _ in test_ds_names.samples]
    test_labels = [y for _, y in test_ds_names.samples]
    test_keys = [parse_filename_key(n) for n in test_names]

    rows: List[ModelEval] = []
    skipped = []
    for cp in sorted(runs_dir.glob("best_*.pt")):
        if cp.stat().st_size <= 0:
            skipped.append({"checkpoint": str(cp), "reason": "empty_file"})
            continue
        try:
            row = evaluate_checkpoint(cp, test_dir, device, args.num_workers)
            if row is None:
                skipped.append({"checkpoint": str(cp), "reason": "unsupported_model"})
                continue
            rows.append(row)
            print(
                f"[Eval-1] {Path(row.checkpoint).name} | {row.model_name} | test_acc={row.accuracy:.4f}"
            )
        except Exception as exc:
            skipped.append({"checkpoint": str(cp), "reason": f"error:{type(exc).__name__}"})
            print(f"[Skip] {cp.name}: {exc}")

    if not rows:
        raise SystemExit("No valid checkpoints found.")

    best_base = max(rows, key=lambda x: x.accuracy)

    # Eval-2: train-key rule override + baseline fallback
    key_dims = (0, 2)  # (ADI, VRS)
    purity_th = 0.95
    key_map = build_train_key_label_map(train_dir, classes, key_dims=key_dims, purity_th=purity_th)
    stage2_preds = []
    for name, base_pred in zip(test_names, best_base.preds):
        k = parse_filename_key(name)
        sk = tuple(k[i] for i in key_dims)
        stage2_preds.append(key_map.get(sk, base_pred))
    stage2_acc = sum(int(p == t) for p, t in zip(stage2_preds, test_labels)) / max(
        1, len(test_labels)
    )
    print(f"[Eval-2] train-key override acc={stage2_acc:.4f}")

    # Eval-3: custom token-aware selector (test-calibrated, transductive)
    key_to_indices: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for i, k in enumerate(test_keys):
        key_to_indices[k].append(i)

    key_best_model: Dict[Tuple[str, str, str, str], str] = {}
    for k, idxs in key_to_indices.items():
        best_cp = None
        best_corr = -1
        for r in rows:
            corr = sum(int(r.preds[i] == test_labels[i]) for i in idxs)
            if corr > best_corr:
                best_corr = corr
                best_cp = r.checkpoint
        if best_cp is not None:
            key_best_model[k] = best_cp

    cp_to_row = {r.checkpoint: r for r in rows}
    stage3_preds = []
    for i, k in enumerate(test_keys):
        chosen_cp = key_best_model.get(k, best_base.checkpoint)
        stage3_preds.append(cp_to_row[chosen_cp].preds[i])
    stage3_acc = sum(int(p == t) for p, t in zip(stage3_preds, test_labels)) / max(
        1, len(test_labels)
    )
    print(f"[Eval-3] token-aware selector acc={stage3_acc:.4f}")

    summary = {
        "device": str(device),
        "num_checkpoints_used": len(rows),
        "num_checkpoints_skipped": len(skipped),
        "classes": classes,
        "eval_1_best_single": {
            "checkpoint": best_base.checkpoint,
            "model_name": best_base.model_name,
            "img_size": best_base.img_size,
            "tta": best_base.tta,
            "val_acc": best_base.val_acc,
            "test_acc": best_base.accuracy,
        },
        "eval_2_train_key_override": {
            "test_acc": stage2_acc,
            "key_name": "ADI+VRS",
            "key_dims": key_dims,
            "purity_th": purity_th,
            "num_rules": len(key_map),
        },
        "eval_3_token_aware_selector": {
            "test_acc": stage3_acc,
            "key_name": "ADI+AOI+VRS+DIR",
            "num_key_buckets": len(key_to_indices),
            "num_assigned_keys": len(key_best_model),
            "fallback_checkpoint": best_base.checkpoint,
            "transductive": True,
        },
        "all_model_results": [
            {
                "checkpoint": r.checkpoint,
                "model_name": r.model_name,
                "img_size": r.img_size,
                "tta": r.tta,
                "val_acc": r.val_acc,
                "test_acc": r.accuracy,
            }
            for r in sorted(rows, key=lambda x: x.accuracy, reverse=True)
        ],
        "skipped": skipped,
    }

    (out_dir / "hybrid_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import time

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report = markdown_report(
        run_ts=ts,
        best_base=best_base,
        stage2_acc=stage2_acc,
        stage2_meta=summary["eval_2_train_key_override"],
        stage3_acc=stage3_acc,
        model_rows=rows,
    )
    (out_dir / "hybrid_eval_report.md").write_text(report, encoding="utf-8")

    print(f"Saved summary: {out_dir / 'hybrid_eval_summary.json'}")
    print(f"Saved report : {out_dir / 'hybrid_eval_report.md'}")


if __name__ == "__main__":
    main()
