# Which alloy composition, what process parameters? Inferring the recipe from optimized metallic microstructure and texture

[![arXiv](https://img.shields.io/badge/arXiv-2610.08165-b31b1b.svg)](https://arxiv.org/abs/2610.08165)
[![Workshop](https://img.shields.io/badge/NeurIPS%202026-AI4Mat%20Workshop-4b44ce.svg)](https://arxiv.org/abs/2610.08165)

Code for the paper **"Which alloy composition, what process parameters? Inferring the
recipe from optimized metallic microstructure and texture"** by Mahish K. Guru,
Jan Bohlen, Louam Lemjid, Marius Tacke, Roland Aydin and Noomane Ben Khalifa,
accepted at the **AI4Mat workshop at NeurIPS 2026**.

- Preprint: [arXiv:2610.08165](https://arxiv.org/abs/2610.08165)

The paper asks whether alloy development can be run in reverse: given an optimized
microstructure and texture, which alloy composition and which extrusion parameters
produced it? We learn this structure -> composition + process step on 107
magnesium-alloy extrusion conditions across 14 alloys, each with optical
micrographs and X-ray texture measurements. Three descriptors are compared:

1. **Conventional**: statistical grain and texture descriptors
2. **Vision embedding**: latents from a pretrained image encoder (code key: `genai`)
3. **GNN**: a graph neural network over the grain network

Under 5-fold condition-grouped cross-validation, conventional descriptors identify
the correct alloy for 65% of held-out conditions (most-frequent-alloy baseline: 17%),
while the learned embeddings stay below 30%. Process parameters can be recovered
but with more noise. Adding microstructure roughly halves the extrusion-temperature
error compared with composition alone. Because the targets are discrete (a
14-alloy lattice and a catalogue of observed press settings), heads that pick from
known options beat heads that predict free values.

This project closes the ICME loop opened by two prior works:

1. **Acta Materialia 2025** (forward structure -> property): statistical microstructure
   and texture descriptors + XGBoost/GP/MLP regressors for mechanical properties.
2. **Co-PiLOT** (property -> structure, inverse design): constrained physics-informed
   latent optimization in a ViT-H/14 + FM-DiT generative latent space
   ([arXiv:2609.37875](https://arxiv.org/abs/2609.37875)).

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{guru2026recipe,
  title         = {Which alloy composition, what process parameters? Inferring the recipe
                   from optimized metallic microstructure and texture},
  author        = {Guru, Mahish K. and Bohlen, Jan and Lemjid, Louam and Tacke, Marius
                   and Aydin, Roland and Ben Khalifa, Noomane},
  booktitle     = {NeurIPS 2026 Workshop on AI for Accelerated Materials Design (AI4Mat)},
  year          = {2026},
  eprint        = {2610.08165},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.08165}
}
```

## Repository layout

```
configs/           YAML configs (pipelines, heads, baselines, paths)
src/icme_mg/       Python package
  labels/          Label extraction (composition + T + v) -> labels.csv
  pipelines/
    conventional/  Statistical descriptors (GSH, Gram, n-point, grain histograms)
    genai/         Vision-embedding latent extraction (pretrained encoder)
    gnn/           Grain-graph construction (DREAM.3D RVEs + experimental) + GNN encoder
  heads/           Composition and process prediction heads
  models/          Flow head, spline flows, lattice heads, adapters, baselines
  protocols/       5-fold condition-grouped cross-validation
  training/        Trainer, splits, datasets
  evaluation/      Metrics, calibration, head metrics
scripts/           Entry points (build datasets, train, evaluate, plots)
  run/             run_cv.py / aggregate_cv.py: the paper's CV benchmark
results/           Paper results (paper_results.md is the source of truth), tables
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

# 2. Assemble descriptor datasets
python scripts/build_conventional.py --config configs/conventional.yaml
python scripts/extract_genai_latents.py --config configs/genai.yaml   # needs checkpoint
python scripts/build_graphs.py --config configs/gnn.yaml

# 3. Paper benchmark: 5 folds x 3 representations
python scripts/run/run_cv.py

# 4. Rebuild the paper tables from the per-fold reports
python scripts/run/aggregate_cv.py
```

## Evaluation protocol

All reported numbers use 5-fold condition-grouped cross-validation over 107
conditions in 14 alloys. Images from one condition never cross folds. Results are
reported as fold mean +/- std; see `results/paper_results.md`. The earlier
single-split (`random_seed0`) study is superseded and archived in `results/archive/`.

Each task conditions on the other's known inputs: composition heads see the
representation plus the known extrusion temperature, log ram speed and extrusion
ratio; process heads see the representation plus the known element wt% and
extrusion ratio.

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
