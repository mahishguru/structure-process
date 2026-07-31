#!/usr/bin/env python
"""Extract frozen GenAI encoder latents (requires genai_checkpoint in paths.yaml)."""
import sys

from icme_mg.pipelines.genai.extract_latents import main

if __name__ == "__main__":
    main(*sys.argv[1:])
