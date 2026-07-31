#!/usr/bin/env bash
# Queue: wait for the running 150 um generation, then
#   1) re-collect 150 (resume pass, writes manifest_150.csv)
#   2) render 150 codec PNGs
#   3) generate RVEs at 90, 100, 120 um (per-image), rendering PNGs after each
# Every step is resumable and skips work that already exists.
# Graph encoding and GenAI latent extraction are deliberately NOT run here.
set -u
PY=".venv/bin/python -u"
GEN_PID=${1:-0}

if [ "$GEN_PID" -gt 0 ]; then
  while kill -0 "$GEN_PID" 2>/dev/null; do sleep 60; done
  echo "=== running 150 um generation (PID $GEN_PID) finished ==="
fi

step() {  # step <name> <logfile> <script> <config>
  local name=$1 log=$2; shift 2
  echo "=== $name -> $log ==="
  $PY "$@" > "$log" 2>&1
  echo "$name exit: $?"
}

step "resume-collect 150" data/gnn/generate_150.log scripts/generate_rves.py configs/gnn.yaml
step "render codec PNGs (150)" data/genai/render_150.log scripts/render_rve_images.py configs/genai.yaml

for s in 90 100 120; do
  step "generate RVEs ($s um)" data/gnn/generate_${s}.log scripts/generate_rves.py configs/gnn_${s}.yaml
  step "render codec PNGs (thru $s um)" data/genai/render_${s}.log scripts/render_rve_images.py configs/genai.yaml
done

echo "=== queue done ==="
