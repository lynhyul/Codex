# No Filename (All Experts) Report

- Run time: 2026-03-13 13:56:37
- Experts: 10

| Rank | Method | Type | Filename Needed | Test Acc |
|---:|---|---|---|---:|
| 1 | `uniform_all` | ensemble | False | 0.9131 |
| 2 | `uniform_subset_val_best` | ensemble | False | 0.9108 |
| 3 | `scalar_weighted_all` | ensemble | False | 0.9077 |
| 4 | `stack_mm` | multimodal | True | 0.9077 |
| 5 | `stack_img_distill` | meta_distill | False | 0.9069 |
| 6 | `best_t9_timm_b4_320` | expert | False | 0.9046 |
| 7 | `stack_img` | meta | False | 0.9008 |
| 8 | `best_t12_swin_tiny` | expert | False | 0.9000 |
| 9 | `best_t5_effb3` | expert | False | 0.9000 |
| 10 | `best_t11_convnext_small` | expert | False | 0.8969 |
| 11 | `best_t4_effb0` | expert | False | 0.8885 |
| 12 | `best_t2_r34` | expert | False | 0.8785 |
| 13 | `best_t3_r50` | expert | False | 0.8785 |
| 14 | `best_t7_effb0_256` | expert | False | 0.8785 |
| 15 | `best_t6_effb0_lr1e4` | expert | False | 0.8754 |
| 16 | `best_t1_r18` | expert | False | 0.8723 |

- Best no-filename: `uniform_all` (0.9131)
- Oracle upper bound (10 experts): 0.9738
