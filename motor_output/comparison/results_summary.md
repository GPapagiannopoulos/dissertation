## Validation fold

| model | AUPRC | AUROC | Brier | ECE | source |
|---|---|---|---|---|---|
| HYBRID: XGBoost 600 + LoRA 23, 50/50 | 0.1986 | 0.8243 | 0.02946 | 0.0036 | `motor_output/comparison/newgrid/explore_validation.json` |
| XGBoost, 600 rounds | 0.1843 | 0.8178 | 0.02971 | 0.0021 | `motor_output/comparison/newgrid/ensemble_vs_xgboost600.json` |
| XGBoost, 300 rounds (round cap, not converged) | 0.1827 | 0.8151 | 0.02976 | 0.0029 | `motor_output/comparison/newgrid/comparison.json` |
| LoRA ensemble, 12 configs x (by-loss + last) | 0.1821 | — | 0.02979 | 0.0010 | `motor_output/comparison/diversity_ng23.json` |
| LoRA ensemble, 12 configs by loss | 0.1799 | — | 0.02985 | 0.0015 | `motor_output/comparison/diversity_ng12.json` |
| LoRA ensemble, rank ladder x (by-loss + last) | 0.1784 | 0.8073 | 0.02988 | 0.0021 | `motor_output/comparison/newgrid/explore_validation.json` |
| LoRA ensemble, rank ladder r=2..32 by loss | 0.1772 | 0.8067 | 0.02989 | 0.0009 | `motor_output/comparison/newgrid/explore_validation.json` |
| LoRA ensemble, placements x (by-loss + last) | 0.1755 | 0.8034 | 0.02993 | 0.0013 | `motor_output/comparison/newgrid/explore_validation.json` |
| LoRA r=8 q/k/v/ff/o | 0.1730 | 0.8027 | 0.02997 | 0.0013 | `runs/ng-all5-r8` |
| LoRA ensemble, placements q/k/v + ff + o by loss | 0.1729 | 0.8019 | 0.02999 | 0.0012 | `motor_output/comparison/newgrid/explore_validation.json` |
| LoRA r=16 a=64 q/k/v/ff | 0.1712 | 0.8014 | 0.03000 | 0.0009 | `runs/ng-all4-r16` |
| LoRA r=4 a=32 q/k/v/ff | 0.1690 | 0.7998 | 0.03005 | 0.0014 | `runs/ng-all4-r4-a32` |
| LoRA r=32 a=32 q/k/v/ff | 0.1683 | 0.8008 | 0.03008 | 0.0017 | `runs/ng-all4-r32-a32` |
| LoRA r=16 a=32 q/k/v/ff | 0.1682 | 0.7991 | 0.03007 | 0.0019 | `runs/pilot-all4-r16-a32` |
| LoRA r=2 a=32 q/k/v/ff | 0.1679 | 0.7989 | 0.03007 | 0.0011 | `runs/ng-all4-r2-a32` |
| LoRA r=8 a=32 q/k/v/ff | 0.1677 | 0.8004 | 0.03008 | 0.0015 | `runs/ng-all4-r8` |
| LoRA r=32 a=128 q/k/v/ff | 0.1660 | 0.7963 | 0.03011 | 0.0018 | `runs/ng-all4-r32` |
| LoRA r=8 q/k/v | 0.1653 | 0.7944 | 0.03015 | 0.0023 | `runs/ng-qkv-r8` |
| MOTOR full fine-tune, seed 1, 30k | 0.1652 | 0.7955 | 0.03033 | 0.0063 | `runs/ng-aki-seed1` |
| MOTOR full fine-tune, seed 1, paired against it | 0.1652 | 0.7955 | 0.03033 | 0.0062 | `motor_output/comparison/newgrid/comparison.json` |
| MOTOR full fine-tune, seed 0, 15k | 0.1645 | 0.7996 | 0.03016 | 0.0020 | `runs/ng-aki-seed0-15k` |
| LoRA r=8 o only | 0.1633 | 0.7954 | 0.03016 | 0.0010 | `runs/ng-o-r8` |
| MOTOR full fine-tune, seed 0, 30k | 0.1626 | 0.7981 | 0.03017 | 0.0013 | `runs/ng-aki-seed0` |
| LoRA r=8 ff only | 0.1616 | 0.7935 | 0.03020 | 0.0012 | `runs/ng-ff-r8` |
| MOTOR full fine-tune, seed 2, 30k | 0.1595 | 0.7956 | 0.03051 | 0.0113 | `runs/ng-aki-seed2` |
| LoRA r=8 q/v | 0.1595 | 0.7944 | 0.03022 | 0.0020 | `runs/ng-qv-seed1` |
| MOTOR full fine-tune, seed 0, lr 3e-6 | 0.1557 | 0.7913 | 0.03031 | 0.0022 | `runs/ng-aki-lowlr` |
| MOTOR frozen + linear probe | 0.0905 | 0.7321 | 0.03150 | 0.0026 | `probe` |

## Test fold

| arm | members | AUPRC [95% CI] | AUROC [95% CI] | Brier | ECE | p@1% | resamples |
|---|---|---|---|---|---|---|---|
| HYBRID: XGBoost 600 + LoRA 23, 50/50 | 24 | 0.20462 | 0.83047 | 0.02965 | 0.00437 | 0.4067 | 500 |
| XGBoost, 600 rounds | 1 | 0.19019 [0.18045, 0.20035] | 0.82510 [0.81966, 0.83018] | 0.02988 | 0.00300 | 0.3809 | 2,000 |
| LoRA ensemble, 23 (by-loss + last) | 23 | 0.18617 [0.17636, 0.19585] | 0.81509 [0.80952, 0.82049] | 0.02999 | 0.00080 | 0.3775 | 2,000 |
| LoRA ensemble, 12 (all last) | 12 | 0.18492 [0.17509, 0.19481] | 0.81499 [0.80934, 0.82043] | 0.03003 | 0.00267 | 0.3765 | 2,000 |
| LoRA ensemble, rank ladder x (by-loss + last) | 10 | 0.18246 | 0.81295 | 0.03007 | 0.00130 | 0.3718 | 500 |
| LoRA ensemble, rank ladder r=2..32 by loss | 5 | 0.18207 | 0.81224 | 0.03009 | 0.00175 | 0.3684 | 500 |
| MOTOR monolithic, 3 seeds x pre-collapse | 12 | 0.18195 [0.17222, 0.19171] | 0.81043 [0.80480, 0.81584] | 0.03010 | 0.00251 | 0.3694 | 2,000 |
| LoRA ensemble, placements x (by-loss + last) | 6 | 0.17897 | 0.80824 | 0.03014 | 0.00081 | 0.3619 | 500 |
| MOTOR monolithic, 1 run x pre-collapse | 4 | 0.17803 [0.16814, 0.18756] | 0.80757 [0.80200, 0.81296] | 0.03018 | 0.00256 | 0.3616 | 2,000 |
| LoRA r=8 q/k/v/ff/o | 1 | 0.17740 [0.16897, 0.18647] | 0.80839 [0.80218, 0.81345] | 0.03017 | 0.00055 | 0.3606 | 200 |
| LoRA ensemble, placements q/k/v + ff + o by loss | 3 | 0.17670 | 0.80662 | 0.03021 | 0.00155 | 0.3567 | 500 |
| LoRA r=16 a=64 q/k/v/ff | 1 | 0.17412 [0.16526, 0.18279] | 0.80580 [0.79963, 0.81060] | 0.03024 | 0.00082 | 0.3567 | 200 |
| LoRA r=8 a=32 q/k/v/ff | 1 | 0.17375 [0.16553, 0.18374] | 0.80550 [0.79946, 0.81061] | 0.03026 | 0.00123 | 0.3460 | 200 |
| LoRA r=32 a=32 q/k/v/ff | 1 | 0.17367 [0.16551, 0.18375] | 0.80597 [0.80043, 0.81119] | 0.03025 | 0.00132 | 0.3522 | 200 |
| LoRA r=16 a=32 q/k/v/ff | 1 | 0.17350 [0.16578, 0.18210] | 0.80532 [0.79948, 0.81093] | 0.03026 | 0.00075 | 0.3488 | 200 |
| LoRA r=4 a=32 q/k/v/ff | 1 | 0.17147 [0.16443, 0.18074] | 0.80443 [0.79877, 0.80979] | 0.03029 | 0.00100 | 0.3518 | 200 |
| LoRA r=32 a=128 q/k/v/ff | 1 | 0.17087 [0.16338, 0.18015] | 0.80307 [0.79667, 0.80901] | 0.03029 | 0.00119 | 0.3565 | 200 |
| LoRA r=2 a=32 q/k/v/ff | 1 | 0.17032 [0.16171, 0.17906] | 0.80447 [0.79874, 0.80974] | 0.03031 | 0.00117 | 0.3476 | 200 |
| MOTOR monolithic, single checkpoint | 1 | 0.16881 [0.15984, 0.17780] | 0.79985 [0.79399, 0.80515] | 0.03053 | 0.00582 | 0.3471 | 2,000 |
| LoRA r=8 q/k/v | 1 | 0.16844 [0.16090, 0.17742] | 0.79935 [0.79358, 0.80469] | 0.03038 | 0.00148 | 0.3458 | 200 |
| LoRA r=8 o only | 1 | 0.16793 [0.16041, 0.17653] | 0.79916 [0.79283, 0.80462] | 0.03038 | 0.00051 | 0.3379 | 200 |
| MOTOR full fine-tune, seed 0, 30k | 1 | 0.16772 [0.16038, 0.17671] | 0.80150 [0.79592, 0.80643] | 0.03037 | 0.00106 | 0.3421 | 200 |
| LoRA r=8 ff only | 1 | 0.16533 [0.15747, 0.17505] | 0.79822 [0.79225, 0.80283] | 0.03042 | 0.00073 | 0.3432 | 200 |
| LoRA r=8 q/v | 1 | 0.16489 [0.15754, 0.17406] | 0.79909 [0.79356, 0.80437] | 0.03043 | 0.00232 | 0.3384 | 200 |
| MOTOR full fine-tune, seed 2, 30k | 1 | 0.16383 [0.15503, 0.17296] | 0.79838 [0.79252, 0.80355] | 0.03070 | 0.01076 | 0.3356 | 200 |
| MOTOR full fine-tune, seed 0, lr 3e-6 | 1 | 0.15584 [0.14834, 0.16415] | 0.79501 | 0.03061 | 0.00231 | 0.3187 | 200 |
| MOTOR frozen + linear probe | 1 | 0.09793 [0.09279, 0.10320] | 0.73587 [0.72910, 0.74269] | 0.03170 | 0.00277 | 0.1913 | 2,000 |

Intervals rest on different numbers of subject-level resamples, so the column is reported: a band at 200 draws is a ladder's own and is coarser than one at 2,000. Point estimates are unaffected.

### Paired differences, test fold

| comparison | AUPRC delta [95% CI] | AUROC delta [95% CI] | ECE delta [95% CI] |
|---|---|---|---|
| LoRA 23 - monolithic, 3 seeds | +0.00422 [+0.00207, +0.00618] * | +0.00466 [+0.00340, +0.00590] * | -0.00171 [-0.00252, -0.00013] * |
| LoRA 23 - monolithic, 1 run | +0.00813 [+0.00530, +0.01094] * | +0.00752 [+0.00581, +0.00923] * | -0.00176 [-0.00216, -0.00067] * |
| LoRA 23 - monolithic, single checkpoint | +0.01736 [+0.01349, +0.02101] * | +0.01524 [+0.01289, +0.01759] * | -0.00501 [-0.00559, -0.00377] * |
| LoRA 23 - XGBoost, 600 rounds | -0.00403 [-0.00875, +0.00079] | -0.01001 [-0.01339, -0.00676] * | -0.00219 [-0.00261, -0.00115] * |

`*` the interval excludes zero. ECE is an error, so a NEGATIVE
delta favours the left arm; AUPRC and AUROC favour it when POSITIVE.
