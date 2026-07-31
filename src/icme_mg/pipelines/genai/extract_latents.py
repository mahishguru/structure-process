"""Extract ViT-FMDiT-1280 latents (16 spatial tokens x 80-D) for all micrographs.

Imports the `Compressor` encoder from the external encoder-decoder repo, loads the
trained checkpoint (key "encoder"), and runs a single forward pass over an image
folder. Images are mapped to condition_ids by longest-prefix match of the filename
stem against labels.csv, so files named e.g.
`AZ31_extruded_200_0.6_<anything>.png` resolve to condition AZ31_extruded_200_0.6.

Output: data/genai/latents.npz with
    latents:       (N, 16, 80) float32
    condition_ids: (N,) str
    filenames:     (N,) str
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPATIAL_TOKENS = 16
TARGET_DIM = 1280


def load_encoder(genai_repo: str | Path, checkpoint: str | Path,
                 device: str = "cuda") -> torch.nn.Module:
    repo = Path(genai_repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    from encoder_arch_pretrained import Compressor  # noqa: external repo

    enc = Compressor(use_gradient_checkpointing=False, trainable_blocks=0,
                     target_dim=TARGET_DIM, spatial_tokens=SPATIAL_TOKENS)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    enc.load_state_dict(state)
    return enc.to(device).eval()


def match_condition(stem: str, condition_ids: list[str]) -> str | None:
    """Longest condition_id that prefixes the filename stem."""
    best = None
    for cid in condition_ids:
        if stem.startswith(cid) and (best is None or len(cid) > len(best)):
            best = cid
    return best


def _make_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((512, 512),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def extract(image_dir: str | Path, encoder: torch.nn.Module,
            condition_ids: list[str], batch_size: int = 32,
            device: str = "cuda") -> dict[str, np.ndarray]:
    from PIL import Image

    tf = _make_transform()
    files, cids = [], []
    for f in sorted(Path(image_dir).rglob("*")):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        cid = match_condition(f.stem, condition_ids)
        if cid is not None:
            files.append(f)
            cids.append(cid)
    if not files:
        raise FileNotFoundError(
            f"No images under {image_dir} matched any condition_id prefix")

    zs = []
    for i in range(0, len(files), batch_size):
        batch = torch.stack([tf(Image.open(f).convert("RGB"))
                             for f in files[i:i + batch_size]]).to(device)
        z = encoder(batch)                       # (B, 1280) flat spatial latent
        zs.append(z.reshape(len(batch), SPATIAL_TOKENS, -1).float().cpu().numpy())
    return {
        "latents": np.concatenate(zs).astype(np.float32),
        "condition_ids": np.array(cids),
        "filenames": np.array([str(f) for f in files]),
    }


def main(config_path: str = "configs/genai.yaml") -> None:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    if not paths.get("genai_checkpoint"):
        raise SystemExit(
            "genai_checkpoint is not set in configs/paths.yaml. "
            "Transfer the trained vitfmdit_1280 checkpoint from the remote "
            "machine first, then set the path.")

    data_root = Path(paths["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_encoder(paths["genai_repo"], paths["genai_checkpoint"], device)
    out = extract(cfg["image_dir"], encoder, list(labels["condition_id"]),
                  cfg.get("batch_size", 32), device)

    out_dir = data_root / "genai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "latents.npz", **out)
    n_cond = len(set(out["condition_ids"]))
    print(f"Wrote {len(out['latents'])} latents ({n_cond} conditions) "
          f"-> {out_dir / 'latents.npz'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/genai.yaml")
    args = ap.parse_args()
    main(args.config)
