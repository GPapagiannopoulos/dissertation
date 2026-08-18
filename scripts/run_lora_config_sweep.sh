#!/usr/bin/env bash
# Stage 8d: train and rank LoRA members that differ by CONFIGURATION, not seed.
#
# The seed sweep varies initialisation and batch order. This one varies the
# hypothesis class -- which weights carry an adapter, and how much capacity each
# adapter has -- while holding the seed fixed, so any diversity it produces is
# attributable to the configuration alone.
#
# Usage, from the repo root:
#     scripts/run_lora_config_sweep.sh                 # every config, in priority order
#     scripts/run_lora_config_sweep.sh all4-r8 ff-r8   # just these
#     MAX_HOURS=6 scripts/run_lora_config_sweep.sh
#
# Two conventions the configs below depend on:
#
#   alpha = 4r, always. peft scales an adapter's output by alpha/r, so holding alpha
#   fixed while raising r shrinks the effective step and confounds capacity with
#   learning rate. 4 is the released configuration's ratio (32/8).
#
#   MAX_HOURS is a wall-clock stop, and a heavier config is slower per step. The
#   member the ensemble wants is step_014000.pt, so the cap must leave room for a
#   slower run to still reach it: at 4.2h a 15%-slower config stops near step 13,500
#   and produces nothing comparable. 5.0h absorbs that.
#
# Budget at ~1.0 s/step: ~5.0 h of training plus ~0.75 h of scoring per config.
# Each half is skipped independently, so an interrupted sweep resumes rather than
# retraining or skipping a config.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv-modelling/bin/python"
SEED="${SEED:-0}"
STEPS="${STEPS:-30000}"
MAX_HOURS="${MAX_HOURS:-5.0}"
# stride 3 over 2000-step saves lands on 2000, 8000 and 14000, so the member the
# ensemble wants is scored without paying for the whole curve
STRIDE="${STRIDE:-3}"
PREFIX="${PREFIX:-lora-cfg-}"

# name | target modules | r | alpha, in priority order. The control member is the
# existing lora-seed0-30k: same seed, same schedule, q/v at r=8.
CONFIGS=(
    "all4-r8|q_proj k_proj v_proj ff_proj|8|32"
    "ff-r8|ff_proj|8|32"
    "qv-r16|q_proj v_proj|16|64"
    "qv-r32|q_proj v_proj|32|128"
    "o-r8|o_proj|8|32"
    "qv-r4|q_proj v_proj|4|16"
    "all4-r16|q_proj k_proj v_proj ff_proj|16|64"
)

selected=("$@")

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name targets rank alpha <<<"${config}"

    if [ ${#selected[@]} -gt 0 ]; then
        wanted=0
        for candidate in "${selected[@]}"; do
            [ "${candidate}" = "${name}" ] && wanted=1
        done
        [ "${wanted}" -eq 0 ] && continue
    fi

    dest="${ROOT}/motor_output/runs/${PREFIX}${name}"

    if [ -e "${dest}" ]; then
        echo "=== ${name}: ${dest} exists, not retraining"
    else
        echo "=== ${name}: r=${rank} alpha=${alpha} on ${targets}, seed ${SEED} -> ${dest}"
        # shellcheck disable=SC2086
        "${PYTHON}" "${ROOT}/scripts/train_motor_aki_lora.py" \
            --dest "${dest}" --seed "${SEED}" \
            --total-steps "${STEPS}" --max-hours "${MAX_HOURS}" \
            --lora-r "${rank}" --lora-alpha "${alpha}" \
            --lora-targets ${targets}
    fi

    if [ -f "${dest}/selection/checkpoint_ranking.json" ]; then
        echo "=== ${name}: already scored, skipping"
    else
        echo "=== ${name}: scoring checkpoints on the validation fold"
        "${PYTHON}" "${ROOT}/scripts/score_checkpoints.py" \
            --run "${dest}" --stride "${STRIDE}"
    fi

    echo "=== ${name}: done"
done

echo "=== lora config sweep complete"
