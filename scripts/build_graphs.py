#!/usr/bin/env python
"""Build grain-adjacency graphs from DREAM.3D RVEs and experimental masks."""
import sys

from icme_mg.pipelines.gnn.build import main

if __name__ == "__main__":
    main(*sys.argv[1:])
