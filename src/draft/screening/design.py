"""Inverse screening: given height, tasks and (some) fixed modules, find the
proportions that work best.

The search runs over the leg ratio, the thigh share of the leg and the foot
length (any of them the concept states absolutely is held fixed). The arm is
not searched: at every point it is the shortest arm that meets the reach
constraints (``reach.py``), because a longer arm only costs shoulder and elbow
torque. Each point is a full mass-loop evaluation; fixed modules are checked, not
resized.

Free slots are chosen as in the final answer (real catalogue parts, or the
trend with ``free_mode: trend`` for a faster, smoother search).

A point is *feasible* when the loop closes, every fixed module covers every
task (after safety factors), every free slot stays inside the catalogue's
density frontier, and every reach constraint holds. Feasible points are ranked
by ``design.objective``:

* ``mass``     lightest robot
* ``margin``   largest worst-case torque margin of the fixed modules (most robust)
* ``cot``      cheapest walking (electrical cost of transport)
* ``balanced`` mean rank over mass, cost of transport and fixed-module margin

After the grid, a finer grid around the winner refines it, and the winner is
evaluated once more choosing real catalogue parts for the free slots.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .actuators import fixed_parts
from .sizing import ConceptResult, evaluate
from .spec import resolve_absolute, set_dotted, validate
from .sweep import run_sweep

DEFAULT_SEARCH = {
    "body.leg_ratio": {"min": 0.38, "max": 0.52, "step": 0.02},
    "body.thigh_fraction": {"min": 0.46, "max": 0.54, "step": 0.04},
    "body.foot_length_ratio": {"min": 0.14, "max": 0.17, "step": 0.03},
}
#: absolute body key that freezes a search variable
FREEZES = {"body.leg_ratio": "leg_length_m", "body.thigh_fraction": "thigh_m",
           "body.foot_length_ratio": "foot_length_m"}
DEFAULT_REACH = {"floor_pick_height_m": 0.10, "shelf_height_m": 1.60}
OBJECTIVES = ("mass", "margin", "cot", "balanced")


@dataclass
class DesignResult:
    base: dict
    axes: dict
    objective: str
    rows: list
    best_row: Optional[dict]
    best: Optional[ConceptResult]      # catalogue-mode evaluation of the winner
    feasible: bool
    notes: list = field(default_factory=list)
    fixed_slots: list = field(default_factory=list)


def _values(spec_range) -> list:
    """``{min, max, step}`` (or ``[min, max, step]``) -> grid; a list of values as given.

    A three-number list is read as a range only when it is ordered and the step
    is smaller than the span, so ``[0.40, 0.45, 0.50]`` stays three values.
    """
    if isinstance(spec_range, dict):
        lo, hi, st = spec_range["min"], spec_range["max"], spec_range["step"]
    elif (isinstance(spec_range, (list, tuple)) and len(spec_range) == 3
          and spec_range[0] < spec_range[1]
          and 0 < spec_range[2] <= spec_range[1] - spec_range[0] + 1e-9):
        lo, hi, st = spec_range
    else:
        return list(spec_range)
    n = int(round((hi - lo) / st)) + 1
    return [round(lo + i * st, 4) for i in range(n)]


def prepare(base: dict, search: Optional[dict] = None, free_mode: str = "catalog"
            ) -> tuple[dict, dict, list]:
    """The spec the search runs on, the axes, and notes on what was assumed."""
    spec = copy.deepcopy(base)
    notes = []
    r = spec["reach"]
    if not any(r[k] is not None for k in ("floor_pick_height_m", "shelf_height_m",
                                          "table_height_m", "forward_reach_m")):
        r.update(DEFAULT_REACH)
        notes.append("未给够取要求，按默认：地面拾取 0.10 m、货架 1.60 m")
    r["arm_from_reach"] = True
    # Free slots are sized the way the final answer will be (catalogue parts by
    # default), so a point the search accepts stays accepted. `trend` is faster.
    spec["actuators"]["mode"] = free_mode
    axes = {}
    for key, rng in (search or DEFAULT_SEARCH).items():
        frozen = FREEZES.get(key)
        if frozen and spec["body"].get(frozen):
            notes.append(f"{key} 由 body.{frozen} = {spec['body'][frozen]} 给定，不搜索")
            continue
        axes[key] = _values(rng)
    return spec, axes, notes


def _fixed_margin(row: dict, slots: list) -> float:
    if not slots:
        return math.nan
    return min(row.get(f"{s}_margin", -math.inf) for s in slots)


def score(rows: list, objective: str, fixed_slots: list) -> None:
    """Attach ``score`` (lower is better) to every feasible row."""
    ok = [r for r in rows if r.get("feasible")]
    if not ok:
        return
    if objective == "margin" and fixed_slots:
        for r in ok:
            r["score"] = -_fixed_margin(r, fixed_slots)
        return
    if objective == "cot":
        for r in ok:
            r["score"] = r["cot_walk"]
        return
    if objective == "balanced":
        keys = [("mass_kg", 1), ("cot_walk", 1)]
        if fixed_slots:
            for r in ok:
                r["_fm"] = -_fixed_margin(r, fixed_slots)
            keys.append(("_fm", 1))
        ranks = {id(r): 0.0 for r in ok}
        for k, _ in keys:
            order = sorted(ok, key=lambda x: x[k])
            for i, r in enumerate(order):
                ranks[id(r)] += i / max(len(ok) - 1, 1)
        for r in ok:
            r["score"] = ranks[id(r)] / len(keys)
            r.pop("_fm", None)
        return
    for r in ok:
        r["score"] = r["mass_kg"]


def closest(rows: list, fixed_slots: list) -> Optional[dict]:
    """When nothing is feasible: the point whose worst shortfall is smallest."""
    cand = [r for r in rows if "error" not in r and r.get("closed")]
    if not cand:
        return None

    def badness(r):
        fm = _fixed_margin(r, fixed_slots) if fixed_slots else 0.0
        return (-(min(fm, 0.0)) + r.get("n_infeasible_actuators", 0)
                + (0 if r.get("reach_ok", True) else 1.0))
    return min(cand, key=badness)


def _refine_axes(axes: dict, best: dict) -> dict:
    out = {}
    for k, vals in axes.items():
        if len(vals) < 2:
            out[k] = vals
            continue
        step = min(abs(b - a) for a, b in zip(vals, vals[1:])) / 2
        c = best[k]
        lo, hi = min(vals), max(vals)
        out[k] = sorted({round(min(max(c + d, lo), hi), 4) for d in (-step, 0.0, step)})
    return out


def run_design(base: dict, objective: str = "mass", search: Optional[dict] = None,
               refine: bool = True, workers: Optional[int] = None,
               progress: Optional[Callable[[int, int], None]] = None,
               free_mode: str = "catalog") -> DesignResult:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if free_mode not in ("catalog", "trend"):
        raise ValueError("free_mode must be catalog or trend")
    spec, axes, notes = prepare(base, search, free_mode)
    fixed_slots = list(fixed_parts(spec))      # also validates the names early
    rows = run_sweep(spec, axes, workers=workers, progress=progress) if axes else []
    if not axes:
        r = evaluate(spec)
        rows = [{**r.summary()}]
    score(rows, objective, fixed_slots)
    ok = [r for r in rows if r.get("feasible")]
    if refine and axes:
        anchor = min(ok, key=lambda r: r["score"]) if ok else closest(rows, fixed_slots)
        if anchor is not None:
            fine = _refine_axes(axes, anchor)
            seen = {tuple(r.get(k) for k in axes) for r in rows}
            extra_points = [dict(zip(fine, combo))
                            for combo in itertools.product(*fine.values())]
            extra_points = [p for p in extra_points if tuple(p[k] for k in axes) not in seen]
            if extra_points:
                more = run_sweep(spec, {k: sorted({p[k] for p in extra_points}) for k in axes},
                                 workers=workers)
                more = [r for r in more if tuple(r.get(k) for k in axes) not in seen]
                rows += more
                score(rows, objective, fixed_slots)
                ok = [r for r in rows if r.get("feasible")]
    best_row = min(ok, key=lambda r: r["score"]) if ok else closest(rows, fixed_slots)
    feasible = bool(ok)
    if not feasible:
        notes.append("没有方案能同时满足所有固定模组与够取要求；下面给出最接近的方案")
    if best_row is not None:
        for k, vals in axes.items():
            if len(vals) > 1 and best_row[k] in (min(vals), max(vals)):
                notes.append(f"推荐方案的 {k} = {best_row[k]} 落在搜索范围边界，"
                             f"最优点可能在范围之外，可扩大 design.search 再算")
    best = None
    if best_row is not None:
        final = spec
        for k in axes:
            final = set_dotted(final, k, best_row[k])
        final = set_dotted(final, "name", base["name"] + "-best")
        final = set_dotted(final, "actuators.mode", "catalog")
        resolve_absolute(final)
        validate(final)
        best = evaluate(final)
        best_row["catalog_mass_kg"] = round(best.mass, 2)
    return DesignResult(base=spec, axes=axes, objective=objective, rows=rows,
                        best_row=best_row, best=best, feasible=feasible, notes=notes,
                        fixed_slots=fixed_slots)


def dimensions(res: ConceptResult) -> list[tuple[str, float, str]]:
    """The proportions table a designer takes away (name, metres, note)."""
    g = res.concept.geo
    lower = g.ankle_h + g.leg
    return [
        ("身高", g.height, ""),
        ("腿长（髋关节—踝关节）", g.leg, f"= {g.leg / g.height:.3f} × 身高"),
        ("大腿（髋—膝）", g.thigh, f"占腿长 {g.thigh / g.leg:.0%}"),
        ("小腿（膝—踝）", g.shank, ""),
        ("踝高（踝关节—脚底）", g.ankle_h, ""),
        ("髋关节高度（站直）", lower, f"下身 / 上身 = {lower / (g.height - lower):.2f}"),
        ("髋—肩", g.hip_to_shoulder, ""),
        ("肩—头顶", g.head_neck, ""),
        ("上臂（肩—肘）", g.upper_arm, ""),
        ("前臂（肘—腕）", g.forearm, ""),
        ("手（腕—抓取点）", g.hand, ""),
        ("臂长（肩—抓取点）", g.arm, g.arm_note),
        ("脚长 / 脚宽", g.foot_length, f"宽 {g.foot_width:.3f} m，踝前 {g.toe:.3f} m"),
        ("髋宽 / 肩宽（关节中心距）", g.hip_width, f"肩宽 {g.shoulder_width:.3f} m"),
    ]
