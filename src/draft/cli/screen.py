"""Concept-stage screening of a humanoid: joint actuators, height and leg
proportions, battery — before any CAD.

    python scripts/screen_humanoid.py concept --config experiments/screening/concept.yaml
    python scripts/screen_humanoid.py concept --config ... --mode catalog --set body.height_m=1.6
    python scripts/screen_humanoid.py sweep   --config experiments/screening/sweep_height_leg.yaml
    python scripts/screen_humanoid.py benchmark

Output goes to ``generated/screening/<name>/`` (git-ignored): an HTML report,
CSV/JSON tables, and a Draft override YAML for ``scripts/generate_robot.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

from ..paths import repo_root


def _sets(items) -> dict:
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        node = out
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    return out


def _load(args) -> dict:
    from ..screening.spec import load_spec, make_spec
    over = _sets(args.set)
    if getattr(args, "mode", None):
        over.setdefault("actuators", {})["mode"] = args.mode
    return load_spec(args.config, over) if args.config else make_spec(over)


def _outdir(args, name: str) -> Path:
    d = Path(args.out) if args.out else repo_root() / "generated" / "screening" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def write_concept(res, out: Path) -> dict:
    from ..screening import report
    from ..screening.export import write_draft_overrides
    from ..screening.sweep import write_csv
    ov = write_draft_overrides(res, out / "draft_overrides.yaml")
    report.write(out / "report.html", report.concept_report(res, str(ov.relative_to(repo_root()))
                                                             if ov.is_relative_to(repo_root()) else str(ov)))
    req_rows = []
    for s, rq in res.req.items():
        a = res.chosen[s]
        req_rows.append({"slot": s, "count": res.concept.mass.actuator_count[s],
                         "peak_tau_Nm": round(rq.peak_tau, 2), "tau_task": rq.governing("tau"),
                         "peak_omega_rad_s": round(rq.peak_omega, 2), "omega_task": rq.governing("omega"),
                         "peak_power_W": round(rq.peak_power, 1), "thermal_rms_Nm": round(rq.thermal_rms, 2),
                         **{f"chosen_{k}": v for k, v in a.row().items() if k != "slot"}})
    write_csv(req_rows, out / "requirements.csv")
    cand_rows = [{"slot": s, "rank": i + 1, **c.row()} for s, cs in res.candidates.items()
                 for i, c in enumerate(cs)]
    if cand_rows:
        write_csv(cand_rows, out / "catalog_candidates.csv")
    summary = _jsonable({**res.summary(), "mass_breakdown_kg": res.concept.mass.breakdown(),
                         "warnings": res.warnings(), "notes": res.notes,
                         "energy": {"mission_W": res.energy.mission_W, "battery_Wh": res.energy.battery_Wh,
                                    "battery_kg": res.energy.battery_kg, "peak_W": res.energy.peak_W,
                                    "peak_current_A": res.energy.peak_current_A,
                                    "cot_walk": res.energy.cot_walk},
                         "spec": res.spec})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def cmd_concept(args) -> int:
    from ..screening.sizing import evaluate
    spec = _load(args)
    t0 = time.time()
    res = evaluate(spec)
    out = _outdir(args, spec["name"])
    write_concept(res, out)
    status = "closed" if res.closed else "NOT CLOSED"
    print(f"{spec['name']}: {status}, mass {res.mass:.1f} kg, battery {res.energy.battery_Wh:.0f} Wh, "
          f"{len(res.warnings())} warnings, {res.iterations} iterations, {time.time() - t0:.1f} s")
    for s, a in res.chosen.items():
        rq = res.req[s]
        print(f"  {s:15s} {rq.peak_tau:7.1f} Nm {rq.peak_omega:5.1f} rad/s -> {a.name} "
              f"({a.mass:.2f} kg){'' if a.feasible else '  [infeasible]'}")
    print(f"report: {out / 'report.html'}")
    return 0 if res.closed else 2


def cmd_sweep(args) -> int:
    from ..screening import report
    from ..screening.sizing import evaluate
    from ..screening.spec import set_dotted
    from ..screening.sweep import best_by, rank, run_sweep, write_csv
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    axes = raw.get("sweep")
    if not axes:
        raise SystemExit(f"{args.config} has no `sweep:` block (dotted key -> list of values)")
    base = _load(args)
    if args.mode:
        base = set_dotted(base, "actuators.mode", args.mode)
    out = _outdir(args, base["name"])
    t0 = time.time()

    def prog(i, n):
        print(f"\r  {i}/{n} concepts ({time.time() - t0:.0f} s)", end="", flush=True)
    rows = run_sweep(base, axes, workers=args.workers, progress=prog)
    print()
    ranked = rank(rows)
    write_csv(rows, out / "sweep.csv")
    details = []
    if ranked and not args.no_best:
        # one catalogue-part design per value of the first axis (e.g. per height)
        first = list(axes)[0]
        best = best_by(rows, first)
        print(f"catalogue selection for the lightest design at each {first} ...")
        for val, row in list(best.items())[:12]:
            spec = base
            for k in axes:
                spec = set_dotted(spec, k, row[k])
            tag = "-".join(f"{k.split('.')[-1]}={row[k]}" for k in axes)
            spec = set_dotted(spec, "name", f"{base['name']}-best-{tag}")
            spec = set_dotted(spec, "actuators.mode", "catalog")
            res = evaluate(spec)
            d = out / "best" / tag
            write_concept(res, d)
            details.append({"point": {k: row[k] for k in axes}, "trend_row": row,
                            "catalog": res.summary(), "report": str(d.relative_to(out) / "report.html"),
                            "parts": {s_: (a_.name, a_.vendor, a_.mass, a_.feasible)
                                      for s_, a_ in res.chosen.items()}})
            print(f"  {tag}: {res.mass:.1f} kg -> {d / 'report.html'}")
    report.write(out / "sweep_report.html", report.sweep_report(base, axes, rows, ranked, details))
    print(f"{len(ranked)}/{len(rows)} feasible; table: {out / 'sweep.csv'}")
    print(f"report: {out / 'sweep_report.html'}")
    return 0


def cmd_benchmark(args) -> int:
    from ..screening import report
    from ..screening.reference import benchmark
    from ..screening.sizing import evaluate
    from ..screening.sweep import write_csv
    base = _load(args)
    out = _outdir(args, "benchmark")
    rows = benchmark(base, evaluate)
    write_csv(rows, out / "benchmark.csv")
    report.write(out / "benchmark_report.html", report.benchmark_report(rows))
    for r in rows:
        print(f"  {r['robot']:20s} real {r['mass_real_kg']:6.1f} kg  pred {r['mass_pred_kg']:6.1f} kg  "
              f"knee {r.get('knee_real_Nm')} / {r.get('knee_pred_Nm')}")
    print(f"report: {out / 'benchmark_report.html'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("concept", cmd_concept, "screen one concept"),
                            ("sweep", cmd_sweep, "grid sweep from the config's `sweep:` block"),
                            ("benchmark", cmd_benchmark, "check the model against published humanoids")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=(name == "sweep"), help="concept YAML")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override a concept key, e.g. body.height_m=1.6 (repeatable)")
        p.add_argument("--out", help="output directory (default generated/screening/<name>)")
        p.add_argument("--mode", choices=("trend", "catalog"), help="actuator selection mode")
        if name == "sweep":
            p.add_argument("--workers", type=int, default=None, help="parallel processes")
            p.add_argument("--no-best", action="store_true", help="skip the catalogue run of the best design")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
