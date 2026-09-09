"""A per-run record of what was trained, on what, for how long, at what cost.

`log.jsonl` carries the trajectory but not the circumstances: nothing in a run folder
says which learning rate produced it, which commit, which torch, which GPU, or how
long it actually took.

Two entry points:

- `build_manifest(..., provenance="runtime")` is written by the training driver,
  which knows its own arguments exactly.
- `build_manifest(..., provenance="reconstructed")` is what
  `scripts/tools/write_manifest.py` produces for runs that predate this module. It
  recovers everything derivable from the log and the environment and records `null`
  for what it cannot, rather than guessing.

The `lr` column reconstructs the peak rate, the warmup length and the schedule's
horizon, because the ramp is linear and the decay is a cosine over `total_steps`.
`accumulate`, `token_budget`, `epochs` and whether the encoder was compiled are not
derivable from any artifact.
"""

import json
import os
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

STALL_FACTOR = 4.0
"""How many times the median inter-record gap counts as a stall.

Train records land every 50 steps at a near-constant rate, so a gap several times
the median means the run was not computing: a competing job, thermal throttling,
or the data loader blocked.

It does **not** detect a suspend, contrary to what this comment first claimed.
`elapsed_s` comes from `time.perf_counter()`, which on Linux reads
`CLOCK_MONOTONIC` -- and that clock does not advance while the machine is
suspended. So a lid-close is invisible here. That is the right behaviour for a
throughput figure, since the reported seconds-per-step is real compute either
way, but it means `active_clock` and `wall_clock` will agree on a run that slept,
and neither equals the time that passed on a calendar.
"""


def git_state(root: Path) -> dict[str, Any]:
    """The commit a run was produced at, and whether the tree was clean.

    Args:
        root (Path): Any path inside the repository.

    Returns:
        dict[str, Any]: `commit`, `branch` and `dirty`, or nulls if the directory
            is not a git repository or git is unavailable. A missing commit is
            recorded rather than raised -- a manifest that refuses to exist
            because git is absent helps nobody.
    """

    def _run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return done.stdout.strip()

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        # None means "unknown", which is not the same as False
        "dirty": None if status is None else bool(status),
    }


def environment() -> dict[str, Any]:
    """The machine and library versions the run executed against.

    Returns:
        dict[str, Any]: Python, torch, CUDA and GPU details plus host CPU and RAM.
            GPU fields are null when no device is visible, so this is safe to call
            on a machine that has none.
    """
    import torch

    gpu: dict[str, Any] = {"name": None, "memory_mib": None, "capability": None}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "memory_mib": round(properties.total_memory / 2**20),
            "capability": f"{properties.major}.{properties.minor}",
        }

    try:
        host_ram_gb = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1
        )
    except (ValueError, OSError):
        host_ram_gb = None

    # cuDNN is queried, never used: the encoder is GEMMs, not convolutions. The
    # query nonetheless initialises cuDNN and raises if the loader finds a
    # version other than the one torch was built against -- which is the default
    # on AWS's Deep Learning AMI, whose LD_LIBRARY_PATH shadows torch's bundled
    # copy. Recording "unavailable" beats losing a finished run to a metadata
    # field, the same failure mode as the PosixPath serialisation crash.
    try:
        cudnn = torch.backends.cudnn.version()
    except (RuntimeError, OSError) as error:
        cudnn = f"unavailable: {error.__class__.__name__}"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn,
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "host_ram_gb": host_ram_gb,
    }


def _format_duration(seconds: float) -> str:
    """Seconds as `H:MM:SS`."""
    whole = max(0, int(seconds))
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def run_metrics(log_path: Path) -> dict[str, Any]:
    """Everything a finished run's log says about its cost and trajectory.

    Gaps beyond `STALL_FACTOR` times the median inter-record time are reported
    and subtracted, giving an active duration that excludes periods the run was
    not computing. Note what this does NOT cover: a machine suspend is invisible,
    because `elapsed_s` comes from `CLOCK_MONOTONIC`, which stops with the
    machine. The seconds-per-step figure is therefore real compute regardless,
    and neither clock here equals calendar time for a run that slept.

    Args:
        log_path (Path): A run's `log.jsonl`.

    Returns:
        dict[str, Any]: Step count, wall clock, active clock, seconds per step,
            peak VRAM, the evaluation trajectory and any detected stalls.

    Raises:
        FileNotFoundError: If the log is absent.
        ValueError: If it holds no train record, which means the run died before
            its fiftieth step.
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"Expected a run log at {log_path}.")

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    train = [r for r in records if r.get("event") == "train"]
    validation = [r for r in records if r.get("event") == "validation"]
    finished = next((r for r in records if r.get("event") == "finished"), None)

    if not train:
        raise ValueError(
            f"{log_path} holds no train record; the run did not reach step 50."
        )

    gaps = [
        (b["elapsed_s"] - a["elapsed_s"], a["step"], b["step"])
        for a, b in zip(train, train[1:], strict=False)
    ]
    median_gap = statistics.median(g for g, _, _ in gaps) if gaps else 0.0
    stalls = [
        {"from_step": lo, "to_step": hi, "seconds": round(gap, 1)}
        for gap, lo, hi in gaps
        if median_gap and gap > STALL_FACTOR * median_gap
    ]

    wall = train[-1]["elapsed_s"]
    stalled_s = sum(s["seconds"] for s in stalls)
    active = wall - stalled_s

    best = min(validation, key=lambda r: r["loss"]) if validation else None

    return {
        "steps": train[-1]["step"],
        # `.get` because runs predating the stop-reason field still have to
        # produce a manifest -- the oldest one here has a `finished` record with
        # no `reason` in it
        "stop_reason": finished.get("reason", "unknown") if finished else "incomplete",
        "wall_clock_s": round(wall, 1),
        "wall_clock": _format_duration(wall),
        # elapsed time minus any window the run spent not computing; equal to
        # wall_clock when nothing stalled, which is the common case
        "active_clock_s": round(active, 1),
        "active_clock": _format_duration(active),
        "seconds_per_step": round(active / max(1, train[-1]["step"]), 4),
        "peak_gib": max(r["peak_gib"] for r in train),
        "final_train_loss": train[-1]["loss"],
        "evaluations": len(validation),
        "evaluation_labels": validation[0]["n"] if validation else None,
        "best_evaluation": (
            {"step": best["step"], "loss": best["loss"], "auprc": best["auprc"]}
            if best
            else None
        ),
        "suspected_stalls": stalls,
    }


def checkpoint_inventory(run: Path) -> list[dict[str, Any]]:
    """Every checkpoint the run left, with its size.

    Args:
        run (Path): The run folder.

    Returns:
        list[dict[str, Any]]: One entry per `.pt`, sorted by name.
    """
    return [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(run.glob("*.pt"))
    ]


def build_manifest(
    run: Path,
    *,
    parameters: dict[str, Any],
    provenance: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Assembles a run's manifest, tolerating a log that does not exist yet.

    Args:
        run (Path): The run folder.
        parameters (dict[str, Any]): The arguments the run was launched with.
            Values that could not be recovered should be `None`, not omitted, so
            the gap is visible in the file.
        provenance (str): `"runtime"` when the driver knows its own arguments,
            `"reconstructed"` when they were recovered afterwards. This is the
            field a reader should check before trusting `parameters`.
        repo_root (Path | None): Where to read git state from. Defaults to this
            file's repository.

    Returns:
        dict[str, Any]: The manifest.

    Raises:
        ValueError: If `provenance` is neither of the two permitted values --
            an unrecognised one would make `parameters` untrustworthy in a way
            nothing downstream could detect.
    """
    if provenance not in {"runtime", "reconstructed"}:
        raise ValueError(
            f"provenance is 'runtime' or 'reconstructed', got {provenance!r}."
        )

    root = repo_root or Path(__file__).resolve().parents[4]
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "run": run.name,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": provenance,
        "parameters": parameters,
        "git": git_state(root),
        "environment": environment(),
        "checkpoints": checkpoint_inventory(run),
    }

    try:
        manifest["results"] = run_metrics(run / "log.jsonl")
    except (FileNotFoundError, ValueError) as error:
        # written at run start, before any log exists, and refreshed at the end
        manifest["results"] = {"unavailable": str(error)}

    return manifest


def write_manifest(run: Path, manifest: dict[str, Any]) -> Path:
    """Writes a manifest into its run folder.

    Args:
        run (Path): The run folder.
        manifest (dict[str, Any]): What `build_manifest` returned.

    Returns:
        Path: The file written.
    """
    destination = run / "manifest.json"
    # default=str, because the manifest is written AFTER training: a driver adding a
    # Path-typed argument would otherwise lose the provenance record for a run that
    # had already cost its whole wall clock. A loose repr beats no manifest.
    destination.write_text(json.dumps(manifest, indent=2, default=str))
    return destination
