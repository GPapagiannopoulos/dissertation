#!/usr/bin/env bash
# Stage 8d, reprioritised: LoRA members that differ by WHICH PROJECTIONS carry an
# adapter, with rank explored only on the target set that won.
#
# Supersedes run_lora_config_sweep.sh's ordering. That script put rank on q/v ahead
# of the wider target sets; the first full-fold result inverted the priority:
#
#     q/v      r=8   step 14,000   AUPRC 0.1670
#     q,k,v,ff r=8   step 14,000   AUPRC 0.1744
#
# +0.0074 from the target set alone, about six times the whole seed spread, so rank
# on q/v is measuring the weak axis. The q/v rank runs are kept at the tail rather
# than deleted, and will simply not be reached if the weekend runs out.
#
# Usage, from the repo root:
#     scripts/run_lora_target_sweep.sh                    # every config, in order
#     scripts/run_lora_target_sweep.sh all4-r16 o-r8      # just these
#
# Conventions inherited from the config sweep:
#
#   alpha = 4r, always, so raising r does not also shrink the adapter's output scale.
#
#   MAX_HOURS is a wall-clock stop and a heavier config is slower per step. all4-r8
#   measured 1.10 s/step against q/v's 0.98, and the member the ensemble wants is
#   step_014000.pt. 5.5h reaches it at up to 1.41 s/step, which covers r=32 over
#   five projections with room to spare.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv-modelling/bin/python"
SEED="${SEED:-0}"
STEPS="${STEPS:-30000}"
MAX_HOURS="${MAX_HOURS:-5.5}"
# stride 3 over 2000-step saves lands on 2000, 8000 and 14000
STRIDE="${STRIDE:-3}"
PREFIX="${PREFIX:-lora-cfg-}"

# name | target modules | r | alpha, in priority order.
CONFIGS=(
    # rank, on the target set that won
    "all4-r16|q_proj k_proj v_proj ff_proj|16|64"
    "all4-r32|q_proj k_proj v_proj ff_proj|32|128"
    # the projections q/v never touches, as candidate ensemble members. Each is
    # expected to be a WEAKER single model; the question is whether it disagrees
    # with the strong ones enough to add to the average
    "o-r8|o_proj|8|32"
    "qkv-r8|q_proj k_proj v_proj|8|32"
    "all5-r8|q_proj k_proj v_proj ff_proj o_proj|8|32"
    # tail: rank on q/v, superseded but kept so spare hours are not wasted
    "qv-r16|q_proj v_proj|16|64"
    "qv-r32|q_proj v_proj|32|128"
    "qv-r4|q_proj v_proj|4|16"
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

echo "=== lora target sweep complete"
