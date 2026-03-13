# Codex: Multi-Modal Automation Pipeline

이 저장소는 `Multi-Modal을 이용한 자동화 프로젝트` 아이디어를 코드로 구현한 버전입니다.

핵심 목표:
- Site 이미지 자동 라벨링(Few-shot + 텍스트 설명)
- 텍스트 기반 라벨 정제(재분류)
- 멀티모달 고성능 Teacher 학습
- Teacher 지식을 이미지 전용 Student로 증류(배포형)

## 1. 구성 파일

- `mm_automation_common.py`
- `mm_automation_fewshot_autolabel.py`
- `mm_automation_train_teacher.py`
- `mm_automation_train_student_distill.py`
- `mm_automation_pipeline.py`
- `class_descriptions_seed_example.json`
- `MM_AUTOMATION_README.md` (상세 버전 문서)

## 2. 파이프라인 개요

1. Few-shot 오토라벨링
- HQ 학습데이터 + 클래스 설명 텍스트를 이용해 Site 이미지에 pseudo label 생성
- 이미지별 자동 설명(`generated_description`) 생성
- confidence 임계치 기반으로 `pseudo_labeled/` 데이터셋 생성

2. 텍스트 기반 라벨 정제
- 텍스트/이미지 임베딩 유사도를 사용해 라벨 재분류
- 변경 여부(`changed_by_refinement`)와 top-k 점수를 로그로 저장

3. Multi-Modal Teacher 학습
- 이미지 인코더 + 텍스트 인코더 결합
- HQ 원본 + pseudo labeled 데이터를 함께 학습
- Epoch별 성능표 및 베스트 체크포인트 저장

4. Student Distillation
- Teacher의 지식을 이미지 전용 Student에 distill
- 운영 환경(텍스트 입력 없음)에서 추론 가능한 모델 생성

## 3. 데이터 구조

기본 구조(예시):
- `train/<class_id>/*.jpg`
- `test/<class_id>/*.jpg`

클래스 설명 JSON 구조:

```json
{
  "103": "Class 103 description ...",
  "105": "Class 105 description ..."
}
```

## 4. 빠른 실행 방법

### A. 전체 파이프라인 한 번에 실행

```bash
python mm_automation_pipeline.py ^
  --hq_train_dir E:/samsung/Dataset/train ^
  --site_dir E:/samsung/Dataset/test ^
  --test_dir E:/samsung/Dataset/test ^
  --class_desc_json E:/samsung/Dataset/class_descriptions_seed_example.json ^
  --out_root E:/samsung/Dataset/runs/mm_pipeline
```

### B. 단계별 실행

1) 오토라벨링 + 정제

```bash
python mm_automation_fewshot_autolabel.py ^
  --hq_train_dir E:/samsung/Dataset/train ^
  --site_dir E:/samsung/Dataset/test ^
  --class_desc_json E:/samsung/Dataset/class_descriptions_seed_example.json ^
  --out_dir E:/samsung/Dataset/runs/mm_autolabel ^
  --freeze_backbone
```

2) Teacher 학습

```bash
python mm_automation_train_teacher.py ^
  --hq_train_dir E:/samsung/Dataset/train ^
  --pseudo_labeled_dir E:/samsung/Dataset/runs/mm_autolabel/pseudo_labeled ^
  --test_dir E:/samsung/Dataset/test ^
  --class_desc_json E:/samsung/Dataset/class_descriptions_seed_example.json ^
  --out_dir E:/samsung/Dataset/runs/mm_teacher
```

3) Student 증류

```bash
python mm_automation_train_student_distill.py ^
  --teacher_ckpt E:/samsung/Dataset/runs/mm_teacher/checkpoints/best_teacher.pt ^
  --train_dir E:/samsung/Dataset/train ^
  --test_dir E:/samsung/Dataset/test ^
  --out_dir E:/samsung/Dataset/runs/mm_student
```

## 5. 주요 출력물

오토라벨링:
- `autolabel_summary.json`
- `auto_labels.csv`
- `auto_descriptions.jsonl`
- `pseudo_labeled/`

Teacher:
- `teacher_report.md`
- `teacher_result.json`
- `checkpoints/best_teacher.pt`

Student:
- `student_report.md`
- `student_result.json`
- `checkpoints/best_student.pt`

## 6. 권장 튜닝 포인트

- `--conf_threshold`: pseudo label 품질/양 트레이드오프
- `--k_shot`: few-shot 프로토타입 안정성
- `--class_desc_json`: 클래스 설명 품질(성능 영향 큼)
- `--freeze_backbone`: 자원 제약(24GB) 환경에서 안정적 학습

## 7. 주의사항

- 클래스 설명 품질이 낮으면 오토라벨링 성능이 빠르게 저하될 수 있습니다.
- 운영 배포는 Student 모델 기준으로 검증하는 것을 권장합니다.
- 불균형 클래스가 크면 class weight/threshold를 함께 조정하세요.
