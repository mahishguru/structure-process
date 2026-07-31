#!/usr/bin/env python
"""Render DREAM.3D RVEs to orientation-codec RGB images for the GenAI encoder."""
import sys

from icme_mg.pipelines.genai.render_rves import main

if __name__ == "__main__":
    main(*sys.argv[1:])
