#!/usr/bin/env python
"""Build per-image conventional descriptor matrices (Acta best combo, 150 um)."""
import sys

from icme_mg.pipelines.conventional.build import main

if __name__ == "__main__":
    main(*sys.argv[1:])
