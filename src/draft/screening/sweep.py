"""Grid sweeps over concept parameters, and how to read them.

A sweep is a mapping of dotted concept keys to value lists; every combination
is one concept, each closed with :func:`sizing.evaluate`. Workers run in
parallel processes (each concept is independent).

:func:`rank` orders the closed, feasible designs (all actuators inside the
density frontier, no failed checks) by total mass, then by electrical cost of
transport — the two numbers that most directly cost money and runtime.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from .sizing import evaluate
from .spec import resolve_absolute, set_dotted, validate


def grid(base: dict, axes: dict) -> list[tuple[dict, dict]]:
    """[(point, spec)] for every combination of ``axes`` (dotted key -> values)."""
    keys = list(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        spec = base
        point = dict(zip(keys, combo))
        for k, v in point.items():
            spec = set_dotted(spec, k, v)
        spec = set_dotted(spec, "name", base["name"] + "-" + "-".join(
            f"{k.split('.')[-1]}={v}" for k, v in point.items()))
        try:
            resolve_absolute(spec)
            validate(spec)
        except ValueError as exc:
            out.append((point, exc))
            continue
        out.append((point, spec))
    return out


def _one(args):
    point, spec = args
    if isinstance(spec, Exception):
        return {**point, "error": str(spec), "closed": False}
    try:
        r = evaluate(spec)
    except Exception as exc:          # a bad corner of the grid must not kill the sweep
        return {**point, "error": f"{type(exc).__name__}: {exc}", "closed": False}
    row = {**point, **r.summary()}
    row["warnings"] = "; ".join(w["name"] for w in r.warnings())
    return row


def run_sweep(base: dict, axes: dict, workers: Optional[int] = None,
              progress: Optional[Callable[[int, int], None]] = None) -> list[dict]:
    jobs = grid(base, axes)
    workers = workers or max(1, (os.cpu_count() or 2) - 0)
    rows = []
    if workers == 1:
        for i, j in enumerate(jobs):
            rows.append(_one(j))
            if progress:
                progress(i + 1, len(jobs))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_one, jobs)):
            rows.append(row)
            if progress:
                progress(i + 1, len(jobs))
    return rows


def rank(rows: list[dict]) -> list[dict]:
    """Feasible designs, lightest first; each flagged ``pareto`` when no other
    design is both lighter and cheaper to walk (lower cost of transport)."""
    ok = [r for r in rows if r.get("feasible") and math.isfinite(r.get("mass_kg", math.inf))]
    for r in ok:
        r["pareto"] = not any(
            o is not r and o["mass_kg"] <= r["mass_kg"] and o["cot_walk"] <= r["cot_walk"]
            and (o["mass_kg"] < r["mass_kg"] or o["cot_walk"] < r["cot_walk"]) for o in ok)
    return sorted(ok, key=lambda r: (r["mass_kg"], r.get("cot_walk", 0)))


def best_by(rows: list[dict], key: str) -> dict:
    """The lightest feasible design for each value of ``key`` (e.g. per height)."""
    out: dict = {}
    for r in rank(rows):
        out.setdefault(r[key], r)
    return dict(sorted(out.items()))


def write_csv(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
