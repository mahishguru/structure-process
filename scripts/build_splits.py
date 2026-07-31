#!/usr/bin/env python
"""Build grouped random condition splits under data/splits/."""
import sys

from icme_mg.training.splits import main

if __name__ == "__main__":
    main(*sys.argv[1:])
