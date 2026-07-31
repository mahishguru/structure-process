#!/usr/bin/env python
"""Generate one DREAM.3D RVE per experimental image (requires PipelineRunner)."""
import sys

from icme_mg.pipelines.gnn.generate_rves import main

if __name__ == "__main__":
    main(*sys.argv[1:])
