#!/usr/bin/env python3
"""Concept-stage screening of a humanoid (actuators, proportions, battery).

    python scripts/screen_humanoid.py concept   --config experiments/screening/concept.yaml
    python scripts/screen_humanoid.py concept   --config experiments/screening/concept.yaml --mode catalog
    python scripts/screen_humanoid.py sweep     --config experiments/screening/sweep_height_leg.yaml
    python scripts/screen_humanoid.py benchmark

Needs only numpy, scipy, pyyaml and matplotlib (no MuJoCo, no GPU).
See docs/screening.md.
"""
from __future__ import annotations

import sys

from draft.cli.screen import main

if __name__ == "__main__":
    sys.exit(main())
