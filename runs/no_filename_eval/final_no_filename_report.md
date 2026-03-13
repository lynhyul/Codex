# No-Filename Assumption Report

- Date: 2026-03-13
- Assumption: 테스트 이미지 파일명/토큰 정보 없음 (이미지만 사용 가능)

## Experiment A (5 experts)

Source: `runs/no_filename_eval/no_filename_summary.json`

| Method | Filename Needed | Test Acc |
|---|---|---:|
| uniform_avg (5 experts) | False | 0.9177 |
| scalar_weighted_avg | False | 0.9146 |
| best single expert (t9) | False | 0.9046 |
| img_gate | False | 0.9031 |
| img_gate_distill | False | 0.9046 |
| mm_teacher (reference) | True | 0.9054 |

- Best no-filename: **0.9177**
- Oracle upper bound (selected 5 experts): 0.9615

## Experiment B (10 experts)

Source: `runs/no_filename_eval_all/all_expert_summary.json`

| Method | Filename Needed | Test Acc |
|---|---|---:|
| uniform_all (10 experts) | False | 0.9131 |
| uniform_subset_val_best | False | 0.9108 |
| scalar_weighted_all | False | 0.9077 |
| stack_img | False | 0.9008 |
| stack_img_distill | False | 0.9069 |
| stack_mm (reference) | True | 0.9077 |

- Best no-filename: **0.9131**
- Oracle upper bound (10 experts): 0.9738

## Final Conclusion

- 현재 파일명 없는 조건에서 가장 좋은 실측 성능은 **0.9177 (91.77%)** 입니다.
- 멀티모달(이미지+토큰) 교사/증류는 이번 설정에서 개선 효과가 크지 않았습니다.
- 토큰 기반 룰 라우팅(이전 95.54%)은 파일명 정보가 있을 때만 강력했습니다.

## Next Candidates (No Filename)

1. OOF(out-of-fold) 스태킹 데이터 생성 후 메타분류기 재학습
2. 추가 전문가 모델(새 학습) 확보 + calibration
3. 이미지 기반 라우팅을 위해 backbone feature-level gate 학습
