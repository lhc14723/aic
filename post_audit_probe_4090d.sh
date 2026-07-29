#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/waas/aic

PY=/home/waas/conda-envs/aic-mm/bin/python
LOCK=.post_audit_probe_4090d.lock
PIDFILE=post_audit_probe_4090d.pid
AUDIT_MARKER=outputs/.submission_4090d_clean_audit_complete
LOG=logs_post_audit_probe_4090d.txt

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "Another post-audit probe is already running." >&2
    exit 73
fi
echo "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

log "Waiting for strict final-submission audit marker"
while [ ! -f "$AUDIT_MARKER" ]; do
    sleep 30
done
log "Final submission audit complete; starting in-scene baseline validation"

"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

from ultralytics import YOLO

from aic_mm.models.fusion import TriModalStem  # noqa: F401

checkpoint = Path("outputs/aic_fusion_stage1_4090d_clean/weights/best.pt")
data = Path("artifacts/intrascene_probe/aic_multispectral_intrascene_probe.yaml")
result = YOLO(str(checkpoint), task="detect").val(
    data=str(data),
    imgsz=960,
    batch=4,
    device=0,
    workers=4,
    amp=False,
    max_det=100,
    plots=False,
    verbose=False,
    project="outputs",
    name="aic_intrascene_probe_stage1_baseline",
    exist_ok=True,
)
summary = {
    "checkpoint": str(checkpoint),
    "data": str(data),
    "map50": float(result.box.map50),
    "map50_95": float(result.box.map),
    "per_class_map50_95": [float(value) for value in result.box.maps],
}
destination = Path("outputs/intrascene_probe_stage1_baseline.json")
destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("INTRASCENE_BASELINE", json.dumps(summary))
PY

log "Starting 40-epoch controlled same-scene exposure probe"
PYTHONUNBUFFERED=1 "$PY" -m scripts.train \
    --config configs/train_intrascene_probe_s_4090d.yaml \
    --yes 2>&1 | tee -a logs_train_intrascene_probe_s_4090d.txt

"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import csv
import json
from pathlib import Path

baseline = json.loads(
    Path("outputs/intrascene_probe_stage1_baseline.json").read_text(encoding="utf-8")
)
results_path = Path("outputs/aic_fusion_intrascene_probe_s_4090d/results.csv")
with results_path.open(newline="", encoding="utf-8") as handle:
    rows = [
        {key.strip(): value for key, value in row.items()}
        for row in csv.DictReader(handle)
    ]
if not rows:
    raise SystemExit("INTRASCENE_PROBE_EMPTY_RESULTS")
metric_key = "metrics/mAP50-95(B)"
best = max(rows, key=lambda row: float(row[metric_key]))
summary = {
    "baseline": baseline,
    "trained_best": {
        "epoch": int(float(best["epoch"])),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best[metric_key]),
    },
    "gain_map50_95": float(best[metric_key]) - float(baseline["map50_95"]),
    "results_csv": str(results_path),
}
destination = Path("outputs/intrascene_probe_summary.json")
destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("INTRASCENE_PROBE_COMPLETE", json.dumps(summary))
PY

touch outputs/.intrascene_probe_complete
log "In-scene probe complete"
