"""Paired subject-level bootstrap: by-loss LoRA ensemble vs the monolithic fine-tune.

Roster verified by reproducing diversity_all_configs_byloss.json to six decimals.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from thesis.modelling.evaluation.intervals import (  # noqa: E402
    _draw_rows,
    _subject_groups,
)
from thesis.modelling.evaluation.metrics import binary_metrics  # noqa: E402

RUNS = Path("motor_output/runs")
COMPARISON = Path("motor_output/comparison")
METRICS = ("auprc", "auroc", "precision_at_1pct", "recall_at_1pct", "brier", "ece")
RESAMPLES = 2000

MEMBERS = [
    ("lora-cfg-all4-r8", "step_014000"),
    ("lora-cfg-all4-r16", "step_014000"),
    ("lora-cfg-all4-r16-a32", "step_014000"),
    ("lora-cfg-all4-r32", "step_014000"),
    ("lora-cfg-all4-r32-a32", "step_014000"),
    ("lora-cfg-all5-r8", "step_014000"),
    ("lora-cfg-ff-r8", "last"),
    ("lora-cfg-o-r8", "step_014000"),
    ("lora-cfg-qkv-r8", "step_014000"),
    ("lora-sched-seed1", "step_014000"),
]


def load(path: Path):
    """Loads a checkpoint from a path."""
    with np.load(path) as handle:
        scores, targets, subjects, times = (handle[k] for k in handle.files)
    order = np.lexsort((times, subjects))
    return scores[order], targets[order], subjects[order], times[order]


stack, key, targets, subjects = [], None, None, None
for run, stem in MEMBERS:
    s, t, su, ti = load(RUNS / run / "selection" / f"{stem}_predictions.npz")
    this = np.stack([su, ti])
    if key is None:
        key, targets, subjects = this, t, su
    else:
        assert np.array_equal(key, this) and np.array_equal(targets, t), run
    stack.append(s)

ensemble = np.stack(stack).mean(axis=0)

v3 = json.loads((COMPARISON / "v3_candidates.json").read_text())
mono_row = min(v3, key=lambda r: r["loss"])
mono, t, su, ti = load(
    COMPARISON / f"v3_{Path(mono_row['checkpoint']).stem}_predictions.npz"
)
assert np.array_equal(np.stack([su, ti]), key) and np.array_equal(t, targets)

point = {
    "ensemble": binary_metrics(ensemble, targets),
    "monolithic": binary_metrics(mono, targets),
}
print(f"monolithic checkpoint {mono_row['checkpoint']} (loss {mono_row['loss']:.5f})")
for name, record in point.items():
    print(f"{name:12s} " + "  ".join(f"{m} {record[m]:.5f}" for m in METRICS))

groups = _subject_groups(subjects)
rng = np.random.default_rng(0)
draws = {m: [] for m in METRICS}
for i in range(RESAMPLES):
    rows = _draw_rows(groups, rng)
    left = binary_metrics(ensemble[rows], targets[rows])
    right = binary_metrics(mono[rows], targets[rows])
    for m in METRICS:
        difference = left[m] - right[m]
        if not np.isnan(difference):
            draws[m].append(difference)
    if (i + 1) % 100 == 0:
        print(f"  draw {i + 1}/{RESAMPLES}", flush=True)

deltas = {}
print(
    "\npaired 95% intervals, ensemble - monolithic, subject-level, "
    f"{RESAMPLES} resamples over {len(groups)} subjects"
)
for m in METRICS:
    values = np.array(draws[m])
    observed = point["ensemble"][m] - point["monolithic"][m]
    lo, hi = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    deltas[m] = {
        "value": observed,
        "lo": lo,
        "hi": hi,
        "p_ensemble_wins": float((values > 0).mean()),
    }
    print(
        f"  {m:20s} {observed:+.5f}  [{lo:+.5f}, {hi:+.5f}]  "
        f"wins {deltas[m]['p_ensemble_wins']:.3f}"
    )

Path("/home/george/.claude/jobs/48f45e42/tmp/paired.json").write_text(
    json.dumps(
        {
            "members": [f"{r}/{s}" for r, s in MEMBERS],
            "monolithic_checkpoint": mono_row["checkpoint"],
            "resamples": RESAMPLES,
            "n_subjects": len(groups),
            "point": point,
            "deltas_ensemble_minus_monolithic": deltas,
        },
        indent=2,
    )
)
print("\nwrote paired.json")
