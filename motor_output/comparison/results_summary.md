| model | AUPRC | AUROC | Brier | ECE | source |
|---|---|---|---|---|---|
| XGBoost, 300 rounds (round cap, not converged) | 0.1827 | 0.8151 | 0.02976 | 0.0029 | `motor_output/comparison/newgrid/comparison.json` |
| LoRA ensemble, 12 configs x (by-loss + last) | 0.1821 | — | 0.02979 | 0.0010 | `motor_output/comparison/diversity_ng23.json` |
| LoRA ensemble, 12 configs by loss | 0.1799 | — | 0.02985 | 0.0015 | `motor_output/comparison/diversity_ng12.json` |
| LoRA r=8 q/k/v/ff/o | 0.1730 | 0.8027 | 0.02997 | 0.0013 | `runs/ng-all5-r8` |
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
