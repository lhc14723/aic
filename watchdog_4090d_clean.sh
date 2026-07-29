#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/waas/aic

WATCH_LOG=logs_watchdog_4090d_clean.txt
WATCH_PIDFILE=watchdog_4090d_clean.pid
WATCH_LOCKFILE=.watchdog_4090d_clean.lock
SUP_PIDFILE=supervisor_4090d_clean.pid
SUP_LOG=logs_full_supervisor_4090d_clean.txt

exec 8>"$WATCH_LOCKFILE"
if ! flock -n 8; then
    echo "Another 4090D watchdog is already running." >&2
    exit 73
fi

echo "$$" > "$WATCH_PIDFILE"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$WATCH_LOG"
}

on_exit() {
    local code=$?
    rm -f "$WATCH_PIDFILE"
    log "WATCHDOG_EXIT code=$code"
}
trap on_exit EXIT

pipeline_ready() {
    [ -f outputs/submission_4090d_clean.zip ] &&
        grep -q 'PIPELINE_READY_FOR_AUDIT' "$SUP_LOG" 2>/dev/null
}

supervisor_alive() {
    local pid
    pid=$(cat "$SUP_PIDFILE" 2>/dev/null || true)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

orphan_training_alive() {
    pgrep -f '/home/waas/conda-envs/aic-mm/bin/python -m scripts.train --config configs/.*4090d_clean' \
        >/dev/null 2>&1
}

supervisor_exhausted() {
    tail -n 30 "$SUP_LOG" 2>/dev/null |
        grep -qE '(stage1|highres|final_all) exhausted retries'
}

start_supervisor() {
    log "RECOVERY restarting clean supervisor with unchanged quality configuration"
    setsid nohup ./supervisor_4090d_clean.sh >/dev/null 2>&1 < /dev/null &
}

current_stage() {
    if [ ! -f outputs/aic_fusion_stage1_4090d_clean/.aic_stage_complete ]; then
        printf 'stage1'
    elif [ ! -f outputs/aic_fusion_highres_4090d_clean/.aic_stage_complete ]; then
        printf 'highres'
    elif [ ! -f outputs/aic_fusion_final_all_4090d_clean/.aic_stage_complete ]; then
        printf 'final_all'
    elif [ ! -f outputs/submission_4090d_clean.zip ]; then
        printf 'predict_or_package'
    else
        printf 'audit_pending'
    fi
}

results_path() {
    case "$1" in
        stage1) printf 'outputs/aic_fusion_stage1_4090d_clean/results.csv' ;;
        highres) printf 'outputs/aic_fusion_highres_4090d_clean/results.csv' ;;
        final_all) printf 'outputs/aic_fusion_final_all_4090d_clean/results.csv' ;;
        *) printf '' ;;
    esac
}

log "WATCHDOG_STARTED"

while true; do
    if pipeline_ready; then
        log "PIPELINE_READY outputs/submission_4090d_clean.zip"
        exit 0
    fi

    if ! supervisor_alive; then
        if supervisor_exhausted; then
            log "ALERT supervisor exhausted retries; preserving failure state for investigation"
        elif orphan_training_alive; then
            log "ALERT supervisor absent but training child still alive; refusing duplicate launch"
        else
            start_supervisor
            sleep 10
        fi
    fi

    stage=$(current_stage)
    results=$(results_path "$stage")
    epoch='NA'
    age='NA'
    if [ -n "$results" ] && [ -s "$results" ]; then
        epoch=$(tail -n 1 "$results" | cut -d, -f1)
        age=$(( $(date +%s) - $(stat -c %Y "$results") ))
    fi

    gpu=$(nvidia-smi \
        --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>&1 || true)
    guard_count=$(
        {
            grep -aEh 'QUALITY_GUARD_(NEGATIVE|NONFINITE)_LOSS|Reducing to batch=' \
                logs_*_4090d_clean.txt 2>/dev/null || true
        } | wc -l
    )
    log "HEARTBEAT stage=$stage epoch=$epoch result_age_s=$age guard_events=$guard_count gpu=[$gpu]"

    if [ "$age" != 'NA' ] && [ "$age" -gt 900 ]; then
        log "ALERT no completed epoch for ${age}s"
    fi

    sleep 60
done
