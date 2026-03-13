import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("[run]", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise SystemExit(ret.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq_train_dir", type=str, default="E:/samsung/Dataset/train")
    parser.add_argument("--site_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--test_dir", type=str, default="E:/samsung/Dataset/test")
    parser.add_argument("--class_desc_json", type=str, default="E:/samsung/Dataset/class_descriptions_template.json")
    parser.add_argument("--out_root", type=str, default="E:/samsung/Dataset/runs/mm_pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    auto_dir = out_root / "autolabel"
    teacher_dir = out_root / "teacher"
    student_dir = out_root / "student"

    # 1) Few-shot autolabel + text-guided refinement
    run_cmd(
        [
            sys.executable,
            "mm_automation_fewshot_autolabel.py",
            "--hq_train_dir",
            args.hq_train_dir,
            "--site_dir",
            args.site_dir,
            "--class_desc_json",
            args.class_desc_json,
            "--out_dir",
            str(auto_dir),
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
            "--freeze_backbone",
        ]
    )

    # 2) Multimodal teacher training with auto-labeled data
    run_cmd(
        [
            sys.executable,
            "mm_automation_train_teacher.py",
            "--hq_train_dir",
            args.hq_train_dir,
            "--pseudo_labeled_dir",
            str(auto_dir / "pseudo_labeled"),
            "--test_dir",
            args.test_dir,
            "--class_desc_json",
            args.class_desc_json,
            "--out_dir",
            str(teacher_dir),
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
        ]
    )

    # 3) Distill to image-only student for site deployment
    run_cmd(
        [
            sys.executable,
            "mm_automation_train_student_distill.py",
            "--teacher_ckpt",
            str(teacher_dir / "checkpoints" / "best_teacher.pt"),
            "--train_dir",
            args.hq_train_dir,
            "--test_dir",
            args.test_dir,
            "--out_dir",
            str(student_dir),
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
        ]
    )

    print("pipeline completed")
    print("autolabel summary:", auto_dir / "autolabel_summary.json")
    print("teacher result:", teacher_dir / "teacher_result.json")
    print("student result:", student_dir / "student_result.json")


if __name__ == "__main__":
    main()
