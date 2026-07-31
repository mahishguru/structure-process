#!/usr/bin/env python
"""Build data/labels/labels.csv from the alloy database + data_labels.xlsx."""
import sys

from icme_mg.labels.build_labels import main

if __name__ == "__main__":
    main(*sys.argv[1:])
