#!/usr/bin/env python
"""Build per-image conventional descriptor matrices (Acta best combo, 150 um)."""
import argparse

from icme_mg.pipelines.conventional.build import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conventional.yaml")
    args = parser.parse_args()
    main(args.config)
