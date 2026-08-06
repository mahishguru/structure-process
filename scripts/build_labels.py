#!/usr/bin/env python
"""Build data/labels/labels.csv from the alloy database + data_labels.xlsx."""
import argparse

from icme_mg.labels.build_labels import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    main(args.config)
