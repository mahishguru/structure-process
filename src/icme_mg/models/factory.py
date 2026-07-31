"""Model factory shared by training and evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from .baselines import IndependentMDN
from .flow_head import FlowTransformerHead


def build_head(cfg: dict, kind: str,
               baselines_config: str = "configs/baselines.yaml") -> nn.Module:
    """kind: flow_ar | ar_gaussian | ar_mdn | mdn_independent."""
    if kind == "mdn_independent":
        import yaml
        bcfg = yaml.safe_load(Path(baselines_config).read_text())
        return IndependentMDN(**bcfg["mdn_independent"])
    hp = dict(cfg["head"])
    if kind in ("flow_ar", "spline"):
        hp["token_head"] = "spline"
    elif kind in ("ar_gaussian", "ar_mdn"):
        hp["token_head"] = kind.split("_")[1]
    else:
        raise ValueError(f"unknown head kind: {kind}")
    return FlowTransformerHead(**hp)


LATTICE_HEADS = ("proto_mix", "level_mix")


def build_lattice_head(cfg: dict, kind: str, labels_train_df,
                       global_alloys: list[str], normalizer) -> nn.Module:
    """Lattice-aware heads need the TRAINING label lattice (prototypes /
    levels) and the fitted normalizer. kind: proto_mix | level_mix."""
    import torch

    from .lattice_heads import (LevelMixtureHead, PrototypeMixtureHead,
                                lattice_data_from_labels)
    hp = cfg["head"]
    lat = lattice_data_from_labels(labels_train_df, normalizer)
    if kind == "proto_mix":
        cls_idx = torch.tensor([global_alloys.index(a)
                                for a in lat["alloys_train"]])
        return PrototypeMixtureHead(
            dim=hp["dim"], prototypes_z=lat["prototypes_z"],
            presence_pat=lat["presence_pat"], proto_class_idx=cls_idx,
            n_alloy_classes=hp["n_alloy_classes"],
            flow_bins=hp["flow_bins"], flow_hidden=hp["flow_hidden"])
    if kind == "level_mix":
        return LevelMixtureHead(
            dim=hp["dim"], levels_z=lat["levels_z"],
            n_alloy_classes=hp["n_alloy_classes"],
            flow_bins=hp["flow_bins"], flow_hidden=hp["flow_hidden"])
    raise ValueError(f"unknown lattice head kind: {kind}")
