# ICME-Mg: Inverse Composition-Process Linkages for Extruded Magnesium Alloys

Joint probabilistic prediction of alloy composition and extrusion process parameters
from microstructure and texture descriptors. Target venue: NeurIPS AI4Mat workshop.

This project closes the ICME loop opened by two prior works:

1. **Acta Materialia 2025** (forward structure -> property): statistical microstructure
   and texture descriptors + XGBoost/GP/MLP regressors for mechanical properties.
2. **NeurIPS 2026 submission** (property -> structure, inverse design): ViT-H/14 +
   FM-DiT generative latent space (Co-PiLOT / Meridian).

Here we learn the remaining linkage: **structure -> composition + process**, as a
full joint density `p(composition, process | descriptor)` using an autoregressive
flow-transformer head shared across three descriptor pipelines.

## Repository layout

```
strategy/          Paper story, architecture spec, experiment matrix, data inventory
configs/           YAML configs (pipelines, head, baselines, paths)
src/icme_mg/       Python package
  labels/          Label extraction (composition + T + v) -> labels.csv
  pipelines/
    conventional/  Statistical descriptors (GSH, Gram, n-point, grain histograms)
    genai/         ViT-FMDiT-1280 latent extraction (pretrained encoder)
    gnn/           Grain-graph construction (DREAM.3D RVEs + experimental) + GNN encoder
  models/          Flow-transformer head, spline flows, adapters, baselines
  training/        Trainer, group-aware splits, balanced sampling
  evaluation/      Metrics, calibration, reports
scripts/           Entry points (build datasets, train, evaluate)
paper/             NeurIPS workshop LaTeX skeleton
data/              Generated artifacts (gitignored)
tests/             Unit tests
```

## Quick start

```bash
# 0. Environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Build labels (composition + process parameters per condition)
python scripts/build_labels.py --config configs/paths.yaml

# 2. Build five frozen random folds over conditions, stratified by alloy
python scripts/build_splits.py --config configs/paths.yaml

# 3. Assemble descriptor datasets
python scripts/build_conventional.py --config configs/conventional.yaml
python scripts/extract_genai_latents.py --config configs/genai.yaml   # needs checkpoint
python scripts/build_graphs.py --config configs/gnn.yaml

# 4. Train
python scripts/train.py --config configs/flow_head.yaml --pipeline conventional --split loco_fold0
python scripts/train_baselines.py --config configs/baselines.yaml --split loco_fold0 --kind xgboost

# 5. Evaluate / aggregate
python scripts/evaluate.py --run-dir runs/<run>

# Optional diagnostic only: leaks conditions across train/test by design
python scripts/random_split_benchmark.py

# Best sparse-data head: condition-balanced fusion + validation-selected
# correction for the empirical alloy-frequency prior
python scripts/logit_adjusted_fusion.py
```

The historical `loco_fold*` filenames contain alloy-stratified random folds
over whole conditions. Images from one condition never cross partitions. A
random split over individual images gives perfect composition prediction because
all test conditions also occur in training; it is retained only as a leakage
diagnostic in `runs/summary_random_image_fuse.json`.

The composition head reaches 19.1% present-element WAPE under this grouped
protocol. It gives every condition equal training mass, fuses CatBoost with
cosine kNN, and selects the long-tail prior-correction strength on validation
conditions before evaluating each untouched test fold.

## Prediction targets

Per alloy x condition: elemental wt% (Al, Zn, Mn, Ce, Nd, Y, Gd, Ca), extrusion
temperature T (deg C), extrusion ram speed v (mm/s). Heat-treated conditions are
excluded by default (`include_heat_treated: false` in configs/paths.yaml).

## External data dependencies (not in repo)

- `database_24_06`  raw database: 17 alloy folders x 121 T_v conditions (OM / Property / XRD)
- `training_data`   precomputed descriptors from the Acta 2025 pipeline
- `encoder-decoder-micro-reconstruction`  pretrained ViT-FMDiT-1280 encoder (checkpoint
  pending transfer from remote machine)
- DREAM.3D 6.5 `PipelineRunner` for synthetic RVE generation

Paths are configured centrally in `configs/paths.yaml`.
