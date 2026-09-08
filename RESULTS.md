# Results inventory

Every number intended for the thesis, the artifact it comes from, and the script that
regenerates it. Nothing here is hand-maintained arithmetic: each row points at a JSON on
disk. If a claim is not in this file, it is not reportable.

**Folds.** Validation is 613,712 landmarks over 26,534 subjects; test is 616,193 over
26,526, at a 3.3847% base rate. Every interval resamples **subjects**, never landmarks —
the 12-hourly grid puts ~9 correlated predictions inside one admission.

**Rule.** Each run's checkpoint is chosen by **minimum validation loss on its own ladder**,
never by AUPRC and never at a hardcoded step. Ensembles average all members equally.

All paths below are relative to `motor_output/comparison/newgrid/` unless stated.

---

## 1. The headline tables

| | |
|---|---|
| artifact | `../results_summary.json`, `../results_summary.md` |
| script | `scripts/evaluate/build_results_summary.py` |
| covers | 28 validation rows, 13 test rows, sorted by AUPRC |

The `SINGLE_RUNS` / `ENSEMBLES` rosters are **hardcoded**: a run is invisible to the table
until it is listed there. Both XGBoost fits are present (600 rounds at `build_results_summary.py:66`,
300 at `:67`), so the stale warning in CLAUDE.md about a missing `ng-xgb-seed0-600` no longer
applies.

## 2. The thesis claim, on the test fold

| | |
|---|---|
| artifacts | `paired_lora_vs_monolithic_test.json`, `paired_ens12_vs_mono_single_test.json`, `paired_small_vs_mono_single_test.json`, `paired_ensemble_gaps_test.json` |
| script | `scripts/evaluate/comparison/paired_intervals.py` |

AUPRC, 2,000 subject resamples, seed 0:

| comparison | delta | 95% CI | wins |
|---|---|---|---|
| LoRA `last2` − monolithic, 3 seeds | +0.00422 | [+0.00207, +0.00618] | 2000/2000 |
| LoRA `last2` − monolithic snapshot | +0.00813 | [+0.00530, +0.01094] | 2000/2000 |
| LoRA `last2` − monolithic single | +0.01736 | [+0.01349, +0.02101] | 2000/2000 |
| LoRA ensemble of 12 − monolithic single | +0.01579 | [+0.01164, +0.01966] | — |
| rank ladder, 5 by loss − monolithic single | +0.01326 | [+0.00894, +0.01728] | — |
| placements, 3 by loss − monolithic single | +0.00789 | [+0.00364, +0.01197] | — |

**Every interval clears zero.** This is the claim the thesis rests on and it survives the
test fold.

**Regeneration note.** `paired_ensemble_gaps.py` was a one-off and is deleted. Its output
is reproducible through `paired_intervals.py` with the four arms named explicitly; the
monolithic arm is 3 seeds × steps 6,000/10,000/14,000/18,000 (the `monolithic_window`
key records that window). All 12 bundles are on disk.

## 3. Single models, and whether seed noise explains anything

| | |
|---|---|
| artifacts | `test_fold_pairs.json`, `test_fold_seed_pairs.json` |
| script | `scripts/evaluate/comparison/paired_intervals.py` |

| comparison | AUPRC delta | 95% CI |
|---|---|---|
| mono seed1 − seed0 | +0.00109 | [−0.00366, +0.00620] |
| mono seed1 − seed2 | +0.00498 | [−0.00003, +0.01018] |
| mono seed0 − seed2 | +0.00389 | [−0.00067, +0.00852] |
| LoRA `all5-r8` − mono seed1 | **+0.00859** | [+0.00436, +0.01266] |
| mono seed1 − probe | +0.07088 | [+0.06318, +0.07853] |

**All three seed pairs bracket zero; the LoRA-vs-monolithic gap does not.** That is the
answer to "is this within seed noise" and it needs to be stated in exactly that order.

## 4. Adapter placement

| | |
|---|---|
| artifact | `test_fold_placement.json` |
| script | `scripts/evaluate/comparison/paired_intervals.py` |

| comparison | AUPRC delta | 95% CI |
|---|---|---|
| `all5-r8` − `ff-r8` | +0.01208 | [+0.00814, +0.01608] |
| `all5-r8` − `o-r8` | +0.00947 | [+0.00576, +0.01313] |
| `all5-r8` − `qkv-r8` | +0.00896 | [+0.00522, +0.01285] |
| `all5-r8` − `all4-r8` | +0.00365 | [+0.00013, +0.00738] |
| `qkv-r8` − `o-r8` | +0.00051 | [−0.00320, +0.00407] |

**Wide target sets beat narrow ones decisively; the two narrow sets at matched capacity are
indistinguishable.** The matched-capacity placement claim from the old grid does not
replicate — say so.

## 5. Diversity

| | |
|---|---|
| artifacts | `diversity_ng12_test.json`, `diversity_ng12last_test.json`, `diversity_ng23_test.json`, `diversity_place3_test.json`, `diversity_rank3_test.json`, `pair_ng-*.json` (3), `within_all5r8_{testing,validation}.json` |
| script | `scripts/evaluate/ensembles/measure_diversity.py` |

23-member set on test: flag overlap (Jaccard) **0.4095**, global rank correlation 0.8943,
**alert-region rank correlation 0.5022**.

**Report the alert-region correlation, never the global one** — the global reads ~0.89 for
any pair because ~96.6% of rows are easy negatives, and quoting it beside a 0.41 Jaccard
invites the objection that they contradict each other. A Jaccard of 0.41 means the members
agree on **58%** of the patients they flag (`2J/(1+J)`), not 41%.

Within one run (`ng-all5-r8`), `step_018000` − `last`: AUPRC **+0.01508**
[+0.01145, +0.01858] — the late checkpoint is individually much worse, which is precisely
why it earns its ensemble place.

## 6. Rank ladder and snapshot ensembles

| | |
|---|---|
| artifacts | `snapshot_by_rank.json`, `snapshot_by_rank_test.json`, `rank_ladder_snapshots.json`, `ambiguity_saturation.json` |
| scripts | `scripts/evaluate/ensembles/measure_snapshot_ensembles.py`, `scripts/evaluate/ensembles/plot_saturation.py` |

Test fold, whole-ladder ensembles of 8 against tail-of-3:

| rank | full ladder (8) | tail (3) |
|---|---|---|
| r=2 | 0.17168 | 0.16537 |
| r=16 | 0.17448 | 0.16396 |
| r=32 | **0.17524** | 0.16302 |

**This tests the prediction CLAUDE.md left open** — that a run which barely degrades makes a
worse snapshot member. It holds: full-ladder AUPRC rises with rank while the tail falls.

## 7. Uncertainty decomposition

| | |
|---|---|
| artifacts | `uncertainty_bags_test.json`, `uncertainty_bag_dependence_test.json`, `uncertainty_bootstrap_by_size_test.json`, `ambiguity_deciles_test.json` |
| scripts | `scripts/evaluate/uncertainty/check_uncertainty_signal.py`, `scripts/evaluate/uncertainty/uncertainty_bag_dependence.py` |

LoRA 23 on test: total 0.12263, aleatoric 0.11924, epistemic **0.00339 (2.76%)**.

**Always name the member set.** The epistemic share is a property of the bag, not the data —
it roughly halves on the 12-member set. `uncertainty_bags_test.json` carries the size curve
(100 subsets per size) that makes that explicit.

## 8. Selective prediction

| | |
|---|---|
| artifacts | `selective_prediction.json`, `selective_prediction_test.json`, `greedy_banded_test.json` |
| scripts | `scripts/evaluate/uncertainty/selective_prediction.py`, `scripts/evaluate/ensembles/band_greedy_selection.py` |

Deferral signals, by how much of each is explained by the prediction itself (lower is more
independent): **logit variance 0.283**, mutual information 0.627, raw variance 0.689,
banded variance ~0.

Raw probability-space variance is nearly a function of the score, so it defers the alert
list. Logit variance is the scale-free signal the design called for. Budgets swept at
0.5/1/2/5%, coverage 100% down to 50%, 200 bands.

## 9. Calibration

| | |
|---|---|
| artifacts | `calibration_validation.json`, `calibration_test.json`, `calibration_test_20bin.json` |
| scripts | `scripts/evaluate/clinical/calibration_curves.py`, `scripts/evaluate/clinical/plot_calibration_curves.py` |
| figures | `motor_output/figures/calibration_testing.png` |

## 10. Decision curves

| | |
|---|---|
| artifacts | `decision_curves.json`, `decision_curves_test.json` |
| scripts | `scripts/evaluate/clinical/decision_curves.py`, `scripts/evaluate/clinical/plot_decision_curves.py` |
| figures | `motor_output/figures/newgrid_test/decision_curves.png`, `decision_curve_differences.png` |

The threshold window (5–15%) comes from the intervention, never from the curves. The 1%
alert budget corresponds to a threshold of ~0.25, not 0.5.

## 11. Horizon transfer

| | |
|---|---|
| artifacts | `horizon_72h_lora.json`, `horizon_72h_comparators.json` |
| script | `scripts/evaluate/horizon/score_horizon.py` |

The landmark grid is horizon-invariant, so this needs no GPU — banked scores are re-joined
against relabelled landmarks. **AUPRC is not comparable across horizons** (it rises with
prevalence); use lift or AUROC. Report 72h as the sensitivity analysis; 7d is an
extrapolation limit, because 67.4% of its windows are truncated at discharge.

---

## Not reportable — delete in the cleanup

* **Old-grid runs** (30 directories, 1.75 GB): every `runs/` entry not prefixed `ng-` or
  `pilot-`. Invalidated by the 2026-08-21 landmark grid change.
* **Old-grid comparison JSONs**: the 46 files at the top level of `motor_output/comparison/`,
  superseded by `newgrid/`. Keep `results_summary.{json,md}`.
* `motor_output/stale_analysis/` (20 MB).
* `scripts/sweeps/run_lora_config_sweep.sh` — superseded first ordering of stage 8d.
* Caches: `scripts/**/__pycache__/`, `.pytest_cache/`, `.ruff_cache/`.
* `motor_pipeline_flashcards.apkg` and its builder (already gitignored).
