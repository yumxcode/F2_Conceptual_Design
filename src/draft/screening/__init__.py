"""Concept-stage screening of a humanoid, before CAD (not part of the paper).

From a handful of numbers -- height, leg ratio, the tasks it must do, payload,
runtime -- close the loop between joint demands, actuators, battery and total
mass, and say which actuators (fitted trend or real catalogue parts) cover it.

    spec.py        the concept: defaults, YAML, validation
    body.py        geometry, mass model, planar sagittal/frontal models
    planar.py      planar recursive Newton-Euler inverse dynamics
    tasks.py       reference motions: stand, walk, stairs, squat-lift, run, ...
    actuators.py   requirements per actuator slot; trend sizing; catalogue match
    energy.py      electrical power, battery, cost of transport
    sizing.py      the mass loop (evaluate)
    sweep.py       grid sweeps and ranking
    reference.py   published humanoids from the survey; benchmark
    export.py      parameter overrides for Draft's humanoid generator
    report.py      HTML reports

Entry point: ``scripts/screen_humanoid.py``. Documentation: ``docs/screening.md``.
"""
