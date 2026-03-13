# No-Filename Evaluation Report

- Run time: 2026-03-13 13:44:44
- Device: cuda
- Assumption: test filename is NOT available at inference

## Performance Table

| Rank | Method | Type | Filename Needed | Test Acc |
|---:|---|---|---|---:|
| 1 | `uniform_avg` | ensemble | False | 0.9177 |
| 2 | `scalar_weighted_avg` | ensemble | False | 0.9146 |
| 3 | `mm_teacher` | multimodal | True | 0.9054 |
| 4 | `best_t9_timm_b4_320` | expert | False | 0.9046 |
| 5 | `img_gate_distill` | meta_distill | False | 0.9046 |
| 6 | `img_gate` | meta | False | 0.9031 |
| 7 | `best_t12_swin_tiny` | expert | False | 0.9000 |
| 8 | `best_t5_effb3` | expert | False | 0.9000 |
| 9 | `best_t11_convnext_small` | expert | False | 0.8969 |
| 10 | `best_t4_effb0` | expert | False | 0.8885 |

## Key Points

- Best no-filename method: `uniform_avg` (0.9177)
- Oracle upper bound with selected experts: 0.9615
- `mm_teacher` is multimodal reference (image+token), not deployable when filename is absent.
- `img_gate_distill` is deployable image-only student distilled from multimodal teacher.

