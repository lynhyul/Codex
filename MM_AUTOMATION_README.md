# Multi-Modal Automation (Few-shot + Teacher/Student)

This package implements the idea from `Multi-Modal을 이용한 자동화 프로젝트.pptx`:

1. Few-shot autolabeling on site images using class descriptions.
2. Text-guided relabel refinement.
3. Multimodal teacher training with pseudo-labeled data.
4. Image-only student distillation for deployment.

## Files

- `mm_automation_common.py`
- `mm_automation_fewshot_autolabel.py`
- `mm_automation_train_teacher.py`
- `mm_automation_train_student_distill.py`
- `mm_automation_pipeline.py`
- `class_descriptions_seed_example.json`

## Step 0: Prepare class descriptions

Use your own class descriptions, or start from:

- `class_descriptions_seed_example.json`

Expected format:

```json
{
  "103": "Class 103 description ...",
  "105": "Class 105 description ..."
}
```

## Step 1: Few-shot autolabel + refinement

```bash
python mm_automation_fewshot_autolabel.py ^
  --hq_train_dir E:/samsung/Dataset/train ^
  --site_dir E:/samsung/Dataset/test ^
  --class_desc_json E:/samsung/Dataset/class_descriptions_seed_example.json ^
  --out_dir E:/samsung/Dataset/runs/mm_autolabel ^
  --freeze_backbone
```

Outputs:

- `autolabel_summary.json`
- `auto_labels.csv`
- `auto_descriptions.jsonl`
- `pseudo_labeled/` (confidence-filtered)

## Step 2: Train multimodal teacher

```bash
python mm_automation_train_teacher.py ^
  --hq_train_dir E:/samsung/Dataset/train ^
  --pseudo_labeled_dir E:/samsung/Dataset/runs/mm_autolabel/pseudo_labeled ^
  --test_dir E:/samsung/Dataset/test ^
  --class_desc_json E:/samsung/Dataset/class_descriptions_seed_example.json ^
  --out_dir E:/samsung/Dataset/runs/mm_teacher
```

Outputs:

- `teacher_report.md` (epoch table)
- `teacher_result.json`
- `checkpoints/best_teacher.pt`

## Step 3: Distill image-only student

```bash
python mm_automation_train_student_distill.py ^
  --teacher_ckpt E:/samsung/Dataset/runs/mm_teacher/checkpoints/best_teacher.pt ^
  --train_dir E:/samsung/Dataset/train ^
  --test_dir E:/samsung/Dataset/test ^
  --out_dir E:/samsung/Dataset/runs/mm_student
```

Outputs:

- `student_report.md` (epoch table)
- `student_result.json`
- `checkpoints/best_student.pt`

## One-command pipeline

```bash
python mm_automation_pipeline.py ^
  --hq_train_dir E:/samsung/Dataset/train ^
  --site_dir E:/samsung/Dataset/test ^
  --test_dir E:/samsung/Dataset/test ^
  --class_desc_json E:/samsung/Dataset/class_descriptions_seed_example.json ^
  --out_root E:/samsung/Dataset/runs/mm_pipeline
```

## Notes

- This code is designed for a constrained GPU environment and supports freezing backbones.
- For production quality, refine class descriptions and tune confidence threshold for pseudo labels.
- Teacher/student reports are generated every epoch for tracking and review.
