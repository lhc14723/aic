#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

cd /home/waas/aic

PY=/home/waas/conda-envs/aic-mm/bin/python
SUPLOG=logs_full_supervisor_4090d_clean.txt
PIDFILE=supervisor_4090d_clean.pid
LOCKFILE=.supervisor_4090d_clean.lock

exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "Another 4090D clean supervisor is already running." >&2
    exit 73
fi

echo "$$" > "$PIDFILE"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$SUPLOG"
}

on_exit() {
    local code=$?
    rm -f "$PIDFILE"
    log "SUPERVISOR_EXIT code=$code"
}
trap on_exit EXIT

validate_checkpoint() {
    local checkpoint=$1
    "$PY" - "$checkpoint" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size < 1_000_000:
    raise SystemExit(f"CHECKPOINT_INVALID_FILE {path}")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model = checkpoint.get("ema")
if model is None:
    model = checkpoint.get("model")
if model is None:
    raise SystemExit(f"CHECKPOINT_MISSING_MODEL {path}")
bad = [
    name
    for name, value in model.state_dict().items()
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all())
]
if bad:
    raise SystemExit(f"CHECKPOINT_NONFINITE {path} {bad[:10]}")
print(f"CHECKPOINT_OK {path} epoch={checkpoint.get('epoch')} size={path.stat().st_size}")
PY
}

prepare_resume() {
    local source_checkpoint=$1
    local resume_checkpoint=$2
    validate_checkpoint "$source_checkpoint"
    cp -f "$source_checkpoint" "$resume_checkpoint.tmp"
    mv -f "$resume_checkpoint.tmp" "$resume_checkpoint"
    validate_checkpoint "$resume_checkpoint"
}

stage_complete() {
    local output_dir=$1
    [ -f "$output_dir/.aic_stage_complete" ] &&
        [ -f "$output_dir/weights/best.pt" ] &&
        [ -f "$output_dir/weights/last.pt" ]
}

retry_stage() {
    local label=$1
    local initial_config=$2
    local resume_config=$3
    local stage_log=$4
    local output_dir=$5
    local attempt=0
    local config
    local code
    local log_start

    if stage_complete "$output_dir"; then
        validate_checkpoint "$output_dir/weights/best.pt"
        validate_checkpoint "$output_dir/weights/last.pt"
        log "$label already complete; skipping"
        return 0
    fi

    while [ "$attempt" -lt 8 ]; do
        if [ -f "$output_dir/weights/last.pt" ]; then
            prepare_resume \
                "$output_dir/weights/last.pt" \
                "$output_dir/weights/last_resume.pt"
            config=$resume_config
        else
            config=$initial_config
        fi

        attempt=$((attempt + 1))
        log "$label attempt=$attempt config=$config"
        log_start=$(stat -c %s "$stage_log" 2>/dev/null || printf '0')
        set +e
        PYTHONUNBUFFERED=1 "$PY" -m scripts.train --config "$config" --yes \
            2>&1 | tee -a "$stage_log"
        code=${PIPESTATUS[0]}
        set -e

        if tail -c "+$((log_start + 1))" "$stage_log" |
            grep -qE 'Reducing to batch=|QUALITY_GUARD_(NEGATIVE|NONFINITE)_LOSS'; then
            log "$label quality guard detected an invalid run; refusing silent degradation"
            code=86
        fi

        if [ "$code" -eq 0 ] &&
            [ -f "$output_dir/weights/best.pt" ] &&
            [ -f "$output_dir/weights/last.pt" ]; then
            validate_checkpoint "$output_dir/weights/best.pt"
            validate_checkpoint "$output_dir/weights/last.pt"
            touch "$output_dir/.aic_stage_complete"
            log "$label complete"
            return 0
        fi

        log "$label failed code=$code; retrying unchanged configuration in 30 seconds"
        sleep 30
    done

    log "$label exhausted retries"
    return 1
}

log "4090D clean supervisor started"
validate_checkpoint weights/yolo26s.pt

retry_stage \
    stage1 \
    configs/train_fusion_4090d_clean.yaml \
    configs/train_fusion_4090d_clean_resume.yaml \
    logs_stage1_4090d_clean.txt \
    outputs/aic_fusion_stage1_4090d_clean \
    || exit 51

retry_stage \
    highres \
    configs/finetune_highres_4090d_clean.yaml \
    configs/finetune_highres_4090d_clean_resume.yaml \
    logs_highres_4090d_clean.txt \
    outputs/aic_fusion_highres_4090d_clean \
    || exit 52

retry_stage \
    final_all \
    configs/final_all_data_4090d_clean.yaml \
    configs/final_all_data_4090d_clean_resume.yaml \
    logs_final_all_4090d_clean.txt \
    outputs/aic_fusion_final_all_4090d_clean \
    || exit 53

log "Starting clean final prediction"
PYTHONUNBUFFERED=1 "$PY" -m scripts.predict \
    --config configs/predict_4090d_clean.yaml \
    2>&1 | tee logs_predict_4090d_clean.txt

log "Starting clean final packaging"
"$PY" -m scripts.package_submission \
    --predictions outputs/test_predictions_4090d_clean \
    --output outputs/submission_4090d_clean.zip \
    2>&1 | tee logs_package_submission_4090d_clean.txt

log "PIPELINE_READY_FOR_AUDIT outputs/submission_4090d_clean.zip"
