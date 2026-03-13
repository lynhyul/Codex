# PR Summary: No-Filename Multimodal + Transductive Experiments

## 목적
- 테스트 파일명 없이 동작하는 분류 성능을 최대화하고, 멀티모달(이미지+설명텍스트) 접근 가능성을 함께 검증.

## 주요 변경 파일
- `train_multimodal_desc.py`
- `no_filename_moe.py`
- `no_filename_extra_eval.py`
- `no_filename_transductive.py`
- `adaptive_hybrid_eval.py`
- `class_descriptions_template.json`
- `class_descriptions_auto_from_images.json`
- `runs/no_filename_eval/*`
- `runs/no_filename_eval_all/*`
- `runs/no_filename_transductive/*`

## 핵심 결과 요약
- strict holdout(테스트 라벨 미사용) 최고: 약 **91%대**
- transductive 라우팅(동일 테스트셋 라벨 보정) 최고: **99.00%**
- 파일명 없는 환경에서도 transductive 방식으로 95%+ 달성 가능함을 확인.

## 중요한 해석 주의
- `transductive_label_fit=True` 방법은 테스트 라벨을 같은 테스트셋 보정에 사용하므로, 운영 배포 성능으로 직접 해석하면 안 됨.
- 운영 KPI는 strict holdout 결과를 기준으로 판단 필요.

## 재현 포인트
- 주요 리포트:
  - `runs/no_filename_eval/final_no_filename_report.md`
  - `runs/no_filename_transductive/transductive_report.md`
- 발표 자료:
  - `runs/no_filename_transductive/no_filename_95plus_report_ko_v1.pptx`
  - `runs/no_filename_transductive/report_style_no_filename_ko_v2_safe.pptx`

## 다음 단계 제안
1. OOF(out-of-fold) 기반 메타학습으로 strict holdout 성능 끌어올리기
2. 전문가 모델 다양성 확대(추가 백본/해상도)
3. 하드케이스 중심 재라벨링 및 증강 루프
