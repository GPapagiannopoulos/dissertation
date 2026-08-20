| model | AUPRC | AUROC | Brier | ECE | source |
|---|---|---|---|---|---|
| XGBoost, full | 0.2001 | 0.8221 | 0.03116 | 0.0021 | `motor_output/comparison/comparison.json` |
| LoRA ensemble, every swept config + q/v, best per member | 0.1919 | — | 0.03138 | 0.0030 | `motor_output/comparison/diversity_all_configs_bestper.json` |
| LoRA ensemble, every swept config + q/v, selected by loss | 0.1866 | — | 0.03151 | 0.0020 | `motor_output/comparison/diversity_all_configs_byloss.json` |
| LoRA ensemble, every swept config + q/v, step-matched | 0.1857 | — | 0.03153 | 0.0024 | `motor_output/comparison/diversity_all_configs.json` |
| LoRA ensemble of 3, q/v + all4 + ff | 0.1807 | — | 0.03163 | 0.0012 | `motor_output/comparison/diversity_qv_all4_ff_n3.json` |
| LoRA r=16 q/k/v/ff | 0.1799 | 0.8000 | 0.03217 | 0.0103 | `runs/lora-cfg-all4-r16` |
| LoRA ensemble of 2, q/v + all4 | 0.1792 | — | 0.03165 | 0.0009 | `motor_output/comparison/diversity_qv_all4_n2.json` |
| MOTOR v3 full fine-tune, seed 0 | 0.1791 | 0.7973 | 0.03194 | 0.0077 | `motor_output/comparison/v3_candidates.json` |
| LoRA ensemble of 4, 3 q/v seeds + all4 | 0.1791 | — | 0.03164 | 0.0014 | `motor_output/comparison/diversity_qv3_all4_n4.json` |
| LoRA r=16 a=32 q/k/v/ff | 0.1790 | 0.7989 | 0.03165 | 0.0028 | `runs/lora-cfg-all4-r16-a32` |
| LoRA r=8 q/k/v/ff/o | 0.1789 | 0.8007 | 0.03228 | 0.0124 | `runs/lora-cfg-all5-r8` |
| LoRA ensemble of 2, all4 + ff | 0.1776 | — | 0.03168 | 0.0022 | `motor_output/comparison/diversity_all4_ff_n2.json` |
| LoRA r=32 a=32 q/k/v/ff | 0.1773 | 0.7974 | 0.03169 | 0.0025 | `runs/lora-cfg-all4-r32-a32` |
| LoRA ensemble of 3, seed-only, 30k, best per member | 0.1751 | — | 0.03173 | 0.0026 | `motor_output/comparison/diversity_sched30k_n3_bestper.json` |
| MOTOR v3 full fine-tune, seed 2 | 0.1750 | 0.7983 | 0.03205 | 0.0084 | `runs/aki-seed2` |
| MOTOR v3 full fine-tune, seed 1 | 0.1749 | 0.7981 | 0.03192 | 0.0072 | `runs/aki-seed1` |
| LoRA r=8 o only | 0.1748 | 0.7974 | 0.03217 | 0.0083 | `runs/lora-cfg-o-r8` |
| LoRA ensemble of 3, seed-only, 30k | 0.1746 | — | 0.03174 | 0.0020 | `motor_output/comparison/diversity_sched30k_n3.json` |
| LoRA r=8 q/k/v/ff | 0.1744 | 0.7992 | 0.03174 | 0.0022 | `runs/lora-cfg-all4-r8` |
| LoRA r=8 ff only | 0.1720 | 0.7966 | 0.03188 | 0.0043 | `runs/lora-cfg-ff-r8` |
| LoRA r=32 q/k/v/ff | 0.1706 | 0.7947 | 0.03235 | 0.0143 | `runs/lora-cfg-all4-r32` |
| LoRA r=8 q/k/v | 0.1694 | 0.7931 | 0.03185 | 0.0025 | `runs/lora-cfg-qkv-r8` |
| XGBoost, recency removed | 0.1690 | 0.7971 | 0.03184 | 0.0025 | `motor_output/comparison/ablation_comparison.json` |
| LoRA r=8 q/v, 30k schedule, seed 1 | 0.1686 | 0.7912 | 0.03188 | 0.0036 | `runs/lora-sched-seed1` |
| LoRA ensemble of 3, seed-only, 15k | 0.1683 | — | 0.03187 | 0.0020 | `motor_output/comparison/diversity_seed_only.json` |
| LoRA r=8 q/v, 30k schedule, seed 0 | 0.1661 | 0.7897 | 0.03192 | 0.0029 | `runs/lora-seed0-30k` |
| LoRA ensemble of 3, bagged 0.632 | 0.1651 | — | 0.03204 | 0.0048 | `motor_output/comparison/diversity_bagged.json` |
| LoRA r=8 q/v, 30k schedule, seed 2 | 0.1638 | 0.7891 | 0.03248 | 0.0136 | `runs/lora-sched-seed2` |
| LoRA r=8 q/v, 15k schedule, seed 1 | 0.1628 | 0.7887 | 0.03200 | 0.0027 | `runs/lora-seed1` |
| LoRA r=8 q/v, 15k schedule, seed 0 | 0.1622 | 0.7873 | 0.03205 | 0.0035 | `runs/lora-seed0` |
| LoRA r=8 q/v, 15k schedule, seed 2 | 0.1616 | 0.7880 | 0.03206 | 0.0041 | `runs/lora-seed2` |
| MOTOR v1, bare head | 0.1610 | 0.7833 | 0.03216 | 0.0074 | `motor_output/comparison/comparison.json` |
| LoRA bagged 0.632, seed 11 | 0.1559 | 0.7795 | 0.03241 | 0.0078 | `runs/lora-bag11` |
| LoRA bagged 0.632, seed 10 | 0.1534 | 0.7777 | 0.03240 | 0.0074 | `runs/lora-bag10` |
| LoRA bagged 0.632, seed 12 | 0.1533 | 0.7798 | 0.03249 | 0.0084 | `runs/lora-bag12` |
| MOTOR frozen + linear probe | 0.1085 | 0.7342 | 0.03308 | 0.0011 | `probe` |
