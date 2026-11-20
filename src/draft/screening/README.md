# `draft.screening`

Concept-stage screening of a humanoid, before CAD: from height, proportions,
tasks, payload and runtime, close the loop between joint demands, actuators
(fitted trends or the 114-part catalogue), battery and total mass. Not part of
the paper; it reuses `draft.trends` and the robot survey and changes neither.

| file | what it owns |
|---|---|
| `spec.py` | the concept: every key with its default, YAML loading, validation |
| `body.py` | geometry from height and ratios, the mass model, planar sagittal/frontal trees |
| `planar.py` | planar recursive Newton-Euler inverse dynamics |
| `tasks.py` | reference motions (stand, walk, stairs, squat/stoop lift, run, arm lift, turn, balance) |
| `actuators.py` | per-slot requirements; trend sizing; catalogue matching |
| `energy.py` | electrical power, battery, cost of transport |
| `sizing.py` | `evaluate()`: the mass loop |
| `sweep.py` | grid sweeps, Pareto ranking |
| `reference.py` | published humanoids from the survey; `benchmark()` |
| `export.py` | parameter overrides for `src/draft/robots/humanoid` |
| `report.py` | HTML reports |

Entry point: `scripts/screen_humanoid.py`. Documentation (Chinese):
[`docs/screening.md`](../../../docs/screening.md). Tests: `tests/test_screening.py`.
