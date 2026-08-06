#!/usr/bin/env python
"""Build the frozen random condition split under data/splits/."""
import argparse

from icme_mg.training.splits import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    main(args.config)
