#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/waas/aic

AUDIT_LOG=logs_audit_4090d_clean.txt
AUDIT_PIDFILE=auditor_4090d_clean.pid
AUDIT_LOCKFILE=.auditor_4090d_clean.lock
SUP_LOG=logs_full_supervisor_4090d_clean.txt
ARCHIVE=outputs/submission_4090d_clean.zip
REPORT=outputs/submission_4090d_clean.audit.json
MARKER=outputs/.submission_4090d_clean_audit_complete
PY=/home/waas/conda-envs/aic-mm/bin/python

exec 7>"$AUDIT_LOCKFILE"
if ! flock -n 7; then
    echo "Another 4090D final auditor is already running." >&2
    exit 73
fi

echo "$$" > "$AUDIT_PIDFILE"

on_exit() {
    local code=$?
    rm -f "$AUDIT_PIDFILE"
    printf '%s AUDITOR_EXIT code=%s\n' "$(date --iso-8601=seconds)" "$code" >> "$AUDIT_LOG"
}
trap on_exit EXIT

printf '%s AUDITOR_WAITING\n' "$(date --iso-8601=seconds)" >> "$AUDIT_LOG"
while true; do
    if [ -f "$ARCHIVE" ] &&
        grep -q 'PIPELINE_READY_FOR_AUDIT outputs/submission_4090d_clean.zip' "$SUP_LOG" 2>/dev/null; then
        break
    fi
    sleep 60
done

printf '%s AUDIT_STARTED\n' "$(date --iso-8601=seconds)" >> "$AUDIT_LOG"
"$PY" scripts/audit_submission_strict.py \
    --archive "$ARCHIVE" \
    --manifest artifacts/test_manifest.jsonl \
    --predictions outputs/test_predictions_4090d_clean \
    --report "$REPORT" \
    >> "$AUDIT_LOG" 2>&1

sha256sum "$ARCHIVE" "$REPORT" > outputs/submission_4090d_clean.SHA256SUMS
touch "$MARKER"
printf '%s AUDIT_PASS report=%s\n' "$(date --iso-8601=seconds)" "$REPORT" >> "$AUDIT_LOG"
