#!/usr/bin/env python
"""Train {adapter/encoder + probabilistic head} for one pipeline and one split.

Examples:
    python scripts/train.py --pipeline conventional --split loco_fold0
    python scripts/train.py --pipeline genai --split loco_fold0
    python scripts/train.py --pipeline gnn --split loao_AZ31 --gnn-source synthetic
    python scripts/train.py --pipeline conventional --split loco_fold0 \
        --head mdn_independent
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

from icme_mg.models.adapters import TokenAdapter, VectorAdapter
from icme_mg.models.factory import LATTICE_HEADS, build_head, build_lattice_head
from icme_mg.models.flow_head import count_parameters
from icme_mg.models.label_space import LabelNormalizer
from icme_mg.training.datasets import (BalancedConditionSampler,
                                       ConventionalDataset, GenAILatentDataset,
                                       GrainGraphDataset, load_labels)
from icme_mg.training.trainer import (Trainer, dict_batch_to_cond,
                                      graph_batch_to_cond)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True,
                    choices=["conventional", "genai", "gnn"])
    ap.add_argument("--split", required=True, help="e.g. loco_fold0, loao_AZ31")
    ap.add_argument("--head", default="flow_ar",
                    choices=["flow_ar", "ar_gaussian", "ar_mdn",
                             "mdn_independent", "proto_mix", "level_mix"])
    ap.add_argument("--gnn-source", default="synthetic",
                    choices=["synthetic", "experimental"])
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help="restrict GNN graphs to these image scales, e.g. 90")
    ap.add_argument("--config", default="configs/flow_head.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training.epochs from the config")
    ap.add_argument("--lr", type=float, default=None,
                    help="override training.lr from the config")
    ap.add_argument("--samples-per-condition", type=int, default=None,
                    help="override training.samples_per_condition")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="override training.weight_decay")
    ap.add_argument("--device", default=None,
                    help="override training.device, e.g. cuda or cpu")
    ap.add_argument("--node-dropout", type=float, default=None,
                    help="override gnn_encoder.node_dropout")
    ap.add_argument("--edge-dropout", type=float, default=None,
                    help="override gnn_encoder.edge_dropout")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    data_root = Path(paths["data_root"])
    labels_csv = data_root / "labels" / "labels.csv"
    split = json.loads((data_root / "splits" / f"{args.split}.json").read_text())
    tcfg = cfg["training"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.lr is not None:
        tcfg["lr"] = args.lr
    if args.samples_per_condition is not None:
        tcfg["samples_per_condition"] = args.samples_per_condition
    if args.weight_decay is not None:
        tcfg["weight_decay"] = args.weight_decay
    if args.device is not None:
        tcfg["device"] = args.device

    # ------------------------------------------------------ data + adapter
    dim = cfg["head"]["dim"]
    model = nn.ModuleDict()
    if args.pipeline == "conventional":
        fold_dir = data_root / "conventional" / args.split
        ds_tr = ConventionalDataset(fold_dir, "train", labels_csv)
        ds_va = ConventionalDataset(fold_dir, "val", labels_csv)
        model["adapter"] = VectorAdapter(
            ds_tr.X.shape[1], dim, cfg["adapter"]["conventional"]["n_tokens"])
        batch_to_cond, collate = dict_batch_to_cond, None
        make_loader = DataLoader
    elif args.pipeline == "genai":
        latents = data_root / "genai" / "latents.npz"
        ds_tr = GenAILatentDataset(latents, labels_csv, split["train"])
        ds_va = GenAILatentDataset(latents, labels_csv, split["val"])
        acfg = cfg["adapter"]["genai"]
        model["adapter"] = TokenAdapter(acfg["token_dim"], dim,
                                        acfg["n_tokens"])
        batch_to_cond, collate = dict_batch_to_cond, None
        make_loader = DataLoader
    else:
        from torch_geometric.loader import DataLoader as GeoDataLoader
        from icme_mg.models.gnn_encoder import GrainGraphEncoder
        graph_dir = data_root / "gnn" / args.gnn_source / "graphs"
        ds_tr = GrainGraphDataset(graph_dir, labels_csv, split["train"],
                                  scales=args.scales)
        ds_va = GrainGraphDataset(graph_dir, labels_csv, split["val"],
                                  scales=args.scales)
        gcfg = yaml.safe_load(Path("configs/gnn.yaml").read_text())["gnn_encoder"]
        if args.node_dropout is not None:
            gcfg["node_dropout"] = args.node_dropout
        if args.edge_dropout is not None:
            gcfg["edge_dropout"] = args.edge_dropout
        model["encoder"] = GrainGraphEncoder(
            hidden_dim=gcfg["hidden_dim"], num_layers=gcfg["num_layers"],
            heads=gcfg["heads"], pool_tokens=gcfg["pool_tokens"],
            out_dim=dim, dropout=gcfg["dropout"],
            node_dropout=gcfg.get("node_dropout", 0.0),
            edge_dropout=gcfg.get("edge_dropout", 0.0))
        batch_to_cond = graph_batch_to_cond
        make_loader = GeoDataLoader

    # ---------------------------------------------------------- normalizer
    all_labels = load_labels(labels_csv)
    labels_df = all_labels.loc[split["train"]]
    normalizer = LabelNormalizer.fit(labels_df)
    if args.head in LATTICE_HEADS:
        global_alloys = sorted(all_labels["alloy"].unique())
        model["head"] = build_lattice_head(cfg, args.head, labels_df,
                                           global_alloys, normalizer)
    else:
        model["head"] = build_head(cfg, args.head)
    print(f"parameters: {count_parameters(model):,}")

    out_dir = Path(args.out or
                   f"runs/{args.pipeline}_{args.head}_{args.split}"
                   + (f"_{args.gnn_source}" if args.pipeline == "gnn" else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(out_dir / "normalizer.json")

    # -------------------------------------------------------------- loaders
    sampler = BalancedConditionSampler(
        ds_tr.condition_ids, tcfg["samples_per_condition"], tcfg["seed"])
    dl_tr = make_loader(ds_tr, batch_size=tcfg["batch_size"], sampler=sampler)
    dl_va = make_loader(ds_va, batch_size=tcfg["batch_size"])

    trainer = Trainer(
        model, normalizer, batch_to_cond, out_dir,
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
        epochs=tcfg["epochs"], aux_weight=tcfg["aux_weight"],
        aux_anneal_frac=tcfg["aux_anneal_frac"], ema_decay=tcfg["ema_decay"],
        grad_clip=tcfg["grad_clip"], patience=tcfg["patience"],
        device=tcfg["device"], seed=tcfg["seed"])
    result = trainer.fit(dl_tr, dl_va, sampler)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
