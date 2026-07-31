#!/usr/bin/env python
"""Evaluate a trained probabilistic run on the test split: posterior sampling,
joint-MAP point estimates, per-label metrics, calibration, alloy top-k.

Example:
    python scripts/evaluate.py --run runs/conventional_flow_ar_loco_fold0 \
        --pipeline conventional --split loco_fold0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from icme_mg import ELEMENTS, LABEL_ORDER
from icme_mg.evaluation.calibration import (interval_coverage, pit_histogram,
                                            pit_values, quantile_ece)
from icme_mg.evaluation.metrics import (aggregate_per_condition, alloy_topk,
                                        joint_map_estimate, per_label_metrics,
                                        presence_f1)
from icme_mg.models.factory import LATTICE_HEADS, build_head, build_lattice_head
from icme_mg.models.label_space import LabelNormalizer
from icme_mg.training.datasets import (ConventionalDataset, GenAILatentDataset,
                                       GrainGraphDataset, load_labels)
from icme_mg.training.trainer import dict_batch_to_cond, graph_batch_to_cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pipeline", required=True,
                    choices=["conventional", "genai", "gnn"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--head", default="flow_ar",
                    choices=["flow_ar", "ar_gaussian", "ar_mdn",
                             "mdn_independent", "proto_mix", "level_mix"])
    ap.add_argument("--gnn-source", default="synthetic")
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help="restrict GNN graphs to these image scales, e.g. 90")
    ap.add_argument("--config", default="configs/flow_head.yaml")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="override training.device, e.g. cuda or cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    data_root = Path(paths["data_root"])
    labels_csv = data_root / "labels" / "labels.csv"
    split = json.loads((data_root / "splits" / f"{args.split}.json").read_text())
    run_dir = Path(args.run)
    n_samples = args.n_samples or cfg["inference"]["n_samples"]
    device = args.device or cfg["training"]["device"]

    # ------------------------------------------------------------- dataset
    if args.pipeline == "conventional":
        ds = ConventionalDataset(data_root / "conventional" / args.split,
                                 "test", labels_csv)
        batch_to_cond, loader = dict_batch_to_cond, DataLoader(ds, batch_size=64)
        feature_dim = ds.X.shape[1]
    elif args.pipeline == "genai":
        ds = GenAILatentDataset(data_root / "genai" / "latents.npz",
                                labels_csv, split["test"])
        batch_to_cond, loader = dict_batch_to_cond, DataLoader(ds, batch_size=64)
        feature_dim = None
    else:
        from torch_geometric.loader import DataLoader as GeoDataLoader
        ds = GrainGraphDataset(data_root / "gnn" / args.gnn_source / "graphs",
                               labels_csv, split["test"], scales=args.scales)
        batch_to_cond = graph_batch_to_cond
        loader = GeoDataLoader(ds, batch_size=64)
        feature_dim = None

    # --------------------------------------------------------------- model
    from icme_mg.models.adapters import TokenAdapter, VectorAdapter
    dim = cfg["head"]["dim"]
    model = nn.ModuleDict()
    if args.pipeline == "conventional":
        model["adapter"] = VectorAdapter(
            feature_dim, dim, cfg["adapter"]["conventional"]["n_tokens"])
    elif args.pipeline == "genai":
        acfg = cfg["adapter"]["genai"]
        model["adapter"] = TokenAdapter(acfg["token_dim"], dim, acfg["n_tokens"])
    else:
        from icme_mg.models.gnn_encoder import GrainGraphEncoder
        gcfg = yaml.safe_load(Path("configs/gnn.yaml").read_text())["gnn_encoder"]
        model["encoder"] = GrainGraphEncoder(
            hidden_dim=gcfg["hidden_dim"], num_layers=gcfg["num_layers"],
            heads=gcfg["heads"], pool_tokens=gcfg["pool_tokens"],
            out_dim=dim, dropout=gcfg["dropout"],
            node_dropout=gcfg.get("node_dropout", 0.0),
            edge_dropout=gcfg.get("edge_dropout", 0.0))
    normalizer = LabelNormalizer.load(run_dir / "normalizer.json")
    if args.head in LATTICE_HEADS:
        all_labels = load_labels(labels_csv)
        global_alloys = sorted(all_labels["alloy"].unique())
        model["head"] = build_lattice_head(
            cfg, args.head, all_labels.loc[split["train"]], global_alloys,
            normalizer)
    else:
        model["head"] = build_head(cfg, args.head)
    ckpt = torch.load(run_dir / "best.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # ------------------------------------------------------------ sampling
    all_cids, all_true, all_nll = [], [], []
    all_samples, all_map, all_pres_map = [], [], []
    with torch.no_grad():
        for batch in loader:
            cond, y, present, _ = batch_to_cond(batch, model, device)
            z, log_det = normalizer.normalize(y)
            out = model["head"](z, present, log_det, cond)
            all_nll.append(out["nll"].cpu())
            s = model["head"].sample(cond, n_samples=n_samples)
            y_s = normalizer.denormalize(
                s["z"].view(-1, len(LABEL_ORDER))).view(s["z"].shape)
            y_s[..., :len(ELEMENTS)] *= s["present"].float()
            z_map, pres_map = joint_map_estimate(s)
            y_map = normalizer.denormalize(z_map)
            y_map[:, :len(ELEMENTS)] *= pres_map.float()
            all_samples.append(y_s.cpu())
            all_map.append(y_map.cpu())
            all_pres_map.append(pres_map.cpu())
            all_true.append(y.cpu())
            cids = (batch["condition_id"] if isinstance(batch, dict)
                    else batch.condition_id)
            all_cids.extend([cids] if isinstance(cids, str) else list(cids))

    y_true = torch.cat(all_true).numpy()
    y_map = torch.cat(all_map).numpy()
    pres_map = torch.cat(all_pres_map).numpy().astype(bool)
    samples = torch.cat(all_samples).numpy()
    nll = torch.cat(all_nll).numpy()

    # ----------------------------------------- condition-level aggregation
    cond_ids, y_map_c = aggregate_per_condition(all_cids, y_map)
    _, y_true_c = aggregate_per_condition(all_cids, y_true)
    _, nll_c = aggregate_per_condition(all_cids, nll[:, None])
    _, pres_c = aggregate_per_condition(all_cids, pres_map.astype(float))
    _, samples_c = aggregate_per_condition(all_cids, samples)

    labels_df = load_labels(labels_csv)
    alloys = sorted(labels_df["alloy"].unique())
    nominal = np.stack([
        labels_df[labels_df["alloy"] == a].iloc[0][ELEMENTS].to_numpy(float)
        for a in alloys])
    true_alloys = [labels_df.loc[c, "alloy"] for c in cond_ids]

    pit = pit_values(y_true_c, samples_c)
    report = {
        "split": args.split, "pipeline": args.pipeline, "head": args.head,
        "n_test_conditions": len(cond_ids),
        "joint_nll": float(nll_c.mean()),
        "per_label": per_label_metrics(y_true_c, y_map_c),
        "presence_f1": presence_f1(y_true_c[:, :len(ELEMENTS)] > 0,
                                   pres_c > 0.5),
        "quantile_ece": quantile_ece(pit),
        "pit_histogram": pit_histogram(pit),
        "coverage_90": interval_coverage(y_true_c, samples_c, alpha=0.1),
        "alloy_topk": alloy_topk(y_map_c, nominal, alloys, true_alloys),
    }
    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    np.savez(run_dir / f"predictions_{args.split}.npz",
             condition_ids=cond_ids, y_true=y_true_c, y_map=y_map_c,
             samples=samples_c, nll=nll_c[:, 0])
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("pit_histogram",)}, indent=2))


if __name__ == "__main__":
    main()
