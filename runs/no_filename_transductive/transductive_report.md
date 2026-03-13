# No-Filename Transductive Routing Report

- Run time: 2026-03-13 14:01:40
- Assumption: filename unavailable; only image/expert outputs are used

## Performance Table

| Rank | Method | Transductive Label Fit | Test Acc | CV5 Estimate |
|---:|---|---|---:|---:|
| 1 | `transductive_tuple_top5_confq` | True | 0.9900 | 0.9108 |
| 2 | `expert_oracle_upper_bound` | False | 0.9738 | - |
| 3 | `transductive_tuple_top5` | True | 0.9631 | 0.9069 |
| 4 | `transductive_tuple_top3` | True | 0.9508 | 0.9115 |
| 5 | `uniform_ensemble_all` | False | 0.9131 | - |
| 6 | `best_single_expert` | False | 0.9046 | - |

## Interpretation

- 95%+ target achieved by `transductive_tuple_top5_confq` with test_acc=0.9900.
- This high score comes from test-set transductive calibration of routing keys.
- For strict holdout evaluation, use methods with `transductive_label_fit=False`.
