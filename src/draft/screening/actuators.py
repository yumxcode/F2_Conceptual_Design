"""From task traces to actuator requirements, and from requirements to actuators.

Requirements are per actuator *slot* (``body.actuator_slots``): one motor
type at one place on the robot. A parallel ankle's two motors share pitch and
roll through the linkage (unit lever ratios assumed):

    τ_A = (τ_pitch + τ_roll)/2,   τ_B = (τ_pitch − τ_roll)/2
    ω_A =  ω_pitch + ω_roll,      ω_B =  ω_pitch − ω_roll

which conserves power (τ_A·ω_A + τ_B·ω_B = τ_p·ω_p + τ_r·ω_r).

Two ways to answer a requirement:

* :func:`size_by_trend` — the lightest *hypothetical* actuator Draft's fitted
  trends allow (mass ``0.0653·τ^0.666``, speed ``ω = 228.5·N^-0.938``), choosing
  the reduction, checked against the catalogue's torque- and power-density
  frontier. Continuous in its inputs, so it is what the sweeps use.
* :func:`match_catalog` — real parts from the 114-module catalogue (plus your
  own list), ranked by mass, that cover every sample of every task.

The torque-speed envelope is a current-limited DC motor on a fixed bus:

    τ_avail(ω) = min( τ_peak,  k·τ_peak·(1 − ω/ω_NL) )

``k = actuators.voltage_limit_factor`` is how far the voltage-limited line
reaches above the current limit at standstill. ``k = 1`` is the catalogue's own
straight-line convention (``P_peak = τ·ω/4``, the most conservative); the
default ``k = 2`` keeps full peak torque up to half the no-load speed, which is
closer to how QDD datasheets read. The density frontier is always checked with
the catalogue's convention so it compares like with like.
Continuous torque is not in the catalogue; it is ``τ_peak / peak_to_continuous``
per category unless your own parts state ``tau_cont_Nm``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..trends import motor_fits
from ..trends.feasibility import rotor_inertia
from ..trends.motor_catalog import load_catalog
from .tasks import TaskResult

_LAWS = None


def laws():
    global _LAWS
    if _LAWS is None:
        _LAWS = motor_fits()
    return _LAWS


# ── requirements ───────────────────────────────────────────────────────────────

@dataclass
class TaskDemand:
    task: str
    label: str
    continuous: bool
    tau: np.ndarray            # |τ| samples, all traces of all motors in the slot
    omega: np.ndarray          # |ω| samples, aligned with tau
    power: np.ndarray          # τ·ω (signed), aligned
    rms: float                 # worst motor's RMS torque over this task's cycle

    @property
    def peak_tau(self) -> float:
        return float(self.tau.max()) if self.tau.size else 0.0

    @property
    def peak_omega(self) -> float:
        return float(self.omega.max()) if self.omega.size else 0.0


@dataclass
class SlotRequirement:
    slot: str
    demands: list = field(default_factory=list)   # [TaskDemand]
    thermal_rms: float = 0.0                       # sizing RMS (mission or continuous task)
    thermal_source: str = ""

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.demands:
            return np.zeros(1), np.zeros(1)
        return (np.concatenate([d.tau for d in self.demands]),
                np.concatenate([d.omega for d in self.demands]))

    @property
    def peak_tau(self) -> float:
        return max((d.peak_tau for d in self.demands), default=0.0)

    @property
    def peak_omega(self) -> float:
        return max((d.peak_omega for d in self.demands), default=0.0)

    @property
    def peak_power(self) -> float:
        return max((float(np.abs(d.power).max()) for d in self.demands if d.power.size),
                   default=0.0)

    def governing(self, what: str) -> str:
        key = {"tau": lambda d: d.peak_tau, "omega": lambda d: d.peak_omega,
               "power": lambda d: float(np.abs(d.power).max()) if d.power.size else 0.0}[what]
        if not self.demands:
            return ""
        return max(self.demands, key=key).label


def _motor_traces(traces_by_joint: dict, slot: str, parallel_ankle: bool):
    """Per-motor trace lists for a slot: [[(tau, omega, share), ...], ...]."""
    if slot == "ankle_drive" and parallel_ankle:
        pitch = {tr.tag: tr for tr in traces_by_joint.get("ankle_pitch", [])}
        roll = {tr.tag: tr for tr in traces_by_joint.get("ankle_roll", [])}
        A, B = [], []
        for tag in sorted(set(pitch) | set(roll)):
            p, r = pitch.get(tag), roll.get(tag)
            ref = p or r
            n = len(ref.t)

            def arr(tr, attr):
                return getattr(tr, attr) if tr is not None and len(tr.t) == n else np.zeros(n)
            tp, tr_ = arr(p, "tau"), arr(r, "tau")
            wp, wr = arr(p, "omega"), arr(r, "omega")
            A.append(((tp + tr_) / 2, wp + wr, ref.share))
            B.append(((tp - tr_) / 2, wp - wr, ref.share))
        return [A, B]
    return [[(tr.tau, tr.omega, tr.share) for tr in traces_by_joint.get(slot, [])]]


def requirements(results: list[TaskResult], slots: list[str], spec: dict) -> dict:
    """Collect every task's demand on every slot, and the thermal (RMS) duty."""
    parallel = spec["mechanism"]["ankle"] == "parallel"
    req = {s: SlotRequirement(s) for s in slots}
    for r in results:
        for s in slots:
            motors = _motor_traces(r.traces, s, parallel)
            if not any(motors):
                continue
            taus, oms, pws, rms = [], [], [], 0.0
            for traces in motors:
                if not traces:
                    continue
                tot = sum(sh for _, _, sh in traces)
                ms = sum(sh * float(np.mean(t ** 2)) for t, _, sh in traces) / max(tot, 1.0)
                rms = max(rms, math.sqrt(ms))
                for t, w, _ in traces:
                    taus.append(np.abs(t))
                    oms.append(np.abs(w))
                    pws.append(t * w)
            req[s].demands.append(TaskDemand(
                r.name, r.label, r.continuous, np.concatenate(taus),
                np.concatenate(oms), np.concatenate(pws), rms))
    # agility floor: move at a minimum speed under part of the peak load
    floors = spec["actuators"].get("speed_floor_rad_s") or {}
    load = spec["actuators"].get("speed_floor_load", 0.3)
    for s, rq in req.items():
        w = floors.get(s, 0.0)
        if w > 0 and rq.demands:
            tau = np.array([load * rq.peak_tau, 0.0])
            om = np.array([w, w])
            rq.demands.append(TaskDemand("agility", f"敏捷性下限 {w:g} rad/s", False,
                                         tau, om, tau * om, 0.0))
    # thermal duty: the mission mix, or any continuous task, whichever is worse
    for s, rq in req.items():
        by_task = {d.task: d for d in rq.demands}
        ms = 0.0
        for item in spec["mission"]:
            d = by_task.get(item["task"])
            if d is not None:
                ms += item["share"] * d.rms ** 2
        best, src = math.sqrt(ms), "任务剖面加权"
        for d in rq.demands:
            if d.continuous and d.rms > best:
                best, src = d.rms, f"持续工况：{d.label}"
        rq.thermal_rms, rq.thermal_source = best, src
    return req


# ── the envelope ───────────────────────────────────────────────────────────────

def available_torque(tau_peak: float, omega_nl: float, omega, k: float = 1.0):
    """Torque available at speed ``omega`` (current limit, then voltage line)."""
    return np.minimum(tau_peak, k * tau_peak * (1.0 - np.asarray(omega) / omega_nl))


def envelope_ratio(tau_peak: float, omega_nl: float, tau: np.ndarray,
                   omega: np.ndarray, k: float = 1.0) -> float:
    """min over samples of (available torque at that speed) / (required torque).

    Below 1 means some sample is outside the envelope.
    """
    avail = available_torque(tau_peak, omega_nl, omega, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(tau > 1e-9, avail / tau, np.where(avail >= 0, np.inf, -np.inf))
    return float(np.min(r)) if r.size else math.inf


def _scaled(rq: SlotRequirement, spec: dict):
    a = spec["actuators"]
    tau, om = rq.samples()
    return (tau * a["torque_safety_factor"], om * a["speed_safety_factor"],
            rq.thermal_rms * a["thermal_safety_factor"])


@dataclass
class ActuatorChoice:
    slot: str
    source: str                # trend | catalog | trend-fallback
    name: str
    category: str
    tau_peak: float
    omega_nl: float
    tau_cont: float
    gear: float
    mass: float
    armature: float
    torque_margin: float       # envelope ratio - 1 (after safety factors)
    thermal_margin: float      # tau_cont / required RMS - 1
    feasible: bool
    notes: list = field(default_factory=list)
    vendor: str = ""
    od_mm: Optional[float] = None
    length_mm: Optional[float] = None

    def row(self) -> dict:
        return {"slot": self.slot, "source": self.source, "name": self.name,
                "vendor": self.vendor, "category": self.category,
                "tau_peak_Nm": round(self.tau_peak, 1),
                "omega_nl_rad_s": round(self.omega_nl, 1),
                "tau_cont_Nm": round(self.tau_cont, 1), "gear": round(self.gear, 1),
                "mass_kg": round(self.mass, 3), "armature_kgm2": round(self.armature, 4),
                "torque_margin": round(self.torque_margin, 3),
                "thermal_margin": round(self.thermal_margin, 3),
                "feasible": self.feasible}


def size_by_trend(rq: SlotRequirement, spec: dict) -> ActuatorChoice:
    """Lightest trend actuator covering the requirement, over a reduction grid."""
    L = laws()
    a = spec["actuators"]
    tau, om, th = _scaled(rq, spec)
    ratio = a["peak_to_continuous"].get("default", 2.5)
    level = a["frontier"]
    front = L.frontier["ALL"]
    td_lim = front["torque_density_Nm_per_kg"][level]
    pd_lim = front["power_density_W_per_kg"][level]
    k = a["voltage_limit_factor"]
    best = None
    fallback = None
    for N in np.geomspace(3.0, 200.0, 90):
        w0 = L.speed_coef * N ** L.speed_exp
        if om.max() >= 0.97 * w0:
            continue
        need = np.maximum(tau, tau / np.maximum(k * (1.0 - om / w0), 1e-6))
        tp = max(float(need.max()), ratio * th, 0.5)
        m = L.mass(tp, N)
        td, pd = tp / m, tp * w0 / 4.0 / m
        sized = L.size(tp, N)
        cand = (m, sized["armature"], N, tp, w0, sized)
        violation = max(td / td_lim, pd / pd_lim)
        if violation <= 1.0 and (best is None or cand[:2] < best[:2]):
            best = cand
        if fallback is None or violation < fallback[0]:
            fallback = (violation, cand)
    notes = []
    feasible = best is not None
    if best is None:
        if fallback is None:
            # faster than any reduction allows: report the speed problem
            return ActuatorChoice(rq.slot, "trend", "超出速度趋势", "-", rq.peak_tau, 0, 0, 0,
                                  math.inf, 0, -1, -1, False,
                                  [f"所需转速 {om.max():.1f} rad/s 超出任何减速比的空载转速"])
        viol, best = fallback
        notes.append(f"超出目录 {level} 密度前沿 {viol:.2f} 倍：需要比现有产品更高的扭矩/功率密度")
    m, arm, N, tp, w0, sized = best
    margin = envelope_ratio(tp, w0, tau, om, k) - 1
    tcont = tp / ratio
    return ActuatorChoice(
        rq.slot, "trend", f"趋势模组 {tp:.0f}Nm/N{N:.0f}", "trend", tp, w0, tcont, N, m, arm,
        margin, _thermal_margin(tcont, th), feasible, notes,
        od_mm=2000 * sized["r"], length_mm=1000 * sized["L"])


def _thermal_margin(tcont: float, need: float) -> float:
    return 99.0 if need < 1e-6 else min(tcont / need - 1, 99.0)


# ── catalogue ─────────────────────────────────────────────────────────────────

def _part_armature(e: dict) -> float:
    J, kind, N = e.get("rotor_inertia_kgm2"), e.get("rotor_inertia_kind"), e.get("gear") or 1.0
    if J is None:
        r = (e.get("radius_mm") or 40.0) / 1000.0
        return N * N * rotor_inertia(r)
    return J if kind == "output" else N * N * J


def load_extra_catalog(path: str | Path) -> list[dict]:
    """Your own parts: YAML list or CSV with name, tau_peak_Nm, omega_NL_rad_s,
    mass_kg and optionally vendor, category, gear, tau_cont_Nm, OD_mm, L_mm,
    rotor_inertia_kgm2, rotor_inertia_kind (rotor|output)."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    out = []
    num = ("tau_peak_Nm", "omega_NL_rad_s", "mass_kg", "gear", "tau_cont_Nm", "OD_mm",
           "L_mm", "rotor_inertia_kgm2", "radius_mm")
    for r in rows:
        e = {k: v for k, v in r.items() if v not in ("", None)}
        for k in num:
            if k in e:
                e[k] = float(e[k])
        for k in ("name", "tau_peak_Nm", "omega_NL_rad_s", "mass_kg"):
            if k not in e:
                raise ValueError(f"{p.name}: part {r.get('name', '?')!r} lacks {k}")
        e.setdefault("vendor", "custom")
        e.setdefault("category", "custom")
        if "OD_mm" in e and "radius_mm" not in e:
            e["radius_mm"] = e["OD_mm"] / 2
        e["custom"] = True
        out.append(e)
    return out


def catalog(spec: dict) -> list[dict]:
    rows = [dict(e) for e in load_catalog()
            if e["category"] in spec["actuators"]["categories"]]
    extra = spec["actuators"].get("extra_catalog")
    if extra:
        rows += load_extra_catalog(extra)
    return rows


def _part_choice(e: dict, slot: str, tau, om, th, spec) -> ActuatorChoice:
    a = spec["actuators"]
    ratio = a["peak_to_continuous"].get(e.get("category"), a["peak_to_continuous"].get("default", 2.5))
    tcont = e.get("tau_cont_Nm") or e["tau_peak_Nm"] / ratio
    env = envelope_ratio(e["tau_peak_Nm"], e["omega_NL_rad_s"], tau, om,
                         a["voltage_limit_factor"])
    thm = _thermal_margin(tcont, th)
    feasible = env >= 1.0 and thm >= 0
    notes = [] if e.get("tau_cont_Nm") else [f"持续扭矩按峰值/{ratio:g} 估算"]
    return ActuatorChoice(
        slot, "catalog", e["name"], e.get("category", "-"), e["tau_peak_Nm"],
        e["omega_NL_rad_s"], tcont, e.get("gear") or 0.0, e["mass_kg"],
        _part_armature(e), env - 1, thm, feasible, notes, vendor=e.get("vendor", ""),
        od_mm=e.get("OD_mm"), length_mm=e.get("L_mm"))


def match_catalog(rq: SlotRequirement, spec: dict, parts: Optional[list] = None,
                  extra_requirements: Optional[list] = None) -> list[ActuatorChoice]:
    """Feasible parts for a slot (and any grouped slots), lightest first."""
    parts = parts if parts is not None else catalog(spec)
    reqs = [rq] + list(extra_requirements or [])
    scaled = [_scaled(r, spec) for r in reqs]
    out = []
    for e in parts:
        choices = [_part_choice(e, r.slot, *sc, spec) for r, sc in zip(reqs, scaled)]
        if all(c.feasible for c in choices):
            c = choices[0]
            c.torque_margin = min(x.torque_margin for x in choices)
            c.thermal_margin = min(x.thermal_margin for x in choices)
            out.append(c)
    out.sort(key=lambda c: (c.mass, -c.torque_margin))
    return out


def nearest_misses(rq: SlotRequirement, spec: dict, n: int = 3) -> list[ActuatorChoice]:
    """The parts that come closest when nothing fits (largest worst-case margin)."""
    tau, om, th = _scaled(rq, spec)
    cands = [_part_choice(e, rq.slot, tau, om, th, spec) for e in catalog(spec)]
    cands.sort(key=lambda c: -min(c.torque_margin, c.thermal_margin))
    return cands[:n]


def choose(req: dict, spec: dict, floor_mass: Optional[dict] = None
           ) -> tuple[dict, dict, dict]:
    """Pick one actuator per slot.

    Returns (chosen, trend, candidates): ``chosen`` drives the mass loop (trend or
    catalogue per ``actuators.mode``), ``trend`` is always the trend sizing, and
    ``candidates`` lists the top catalogue parts per slot. ``floor_mass`` makes a
    catalogue choice no lighter than a previous one (so the loop cannot cycle).
    """
    a = spec["actuators"]
    parts = catalog(spec)
    trend = {s: size_by_trend(rq, spec) for s, rq in req.items()}
    groups = {s: [s] for s in req}
    for name, members in (a.get("groups") or {}).items():
        members = [m for m in members if m in req]
        for m in members:
            groups[m] = members
    cands: dict = {}
    full = {}
    for s, rq in req.items():
        others = [req[m] for m in groups[s] if m != s]
        full[s] = match_catalog(rq, spec, parts, others)
        cands[s] = full[s][: max(a["top_n"], 1)]
    chosen = {}
    for s in req:
        if a["mode"] == "catalog":
            if full[s]:
                floor = (floor_mass or {}).get(s, 0.0) - 1e-9
                heavier = [c for c in full[s] if c.mass >= floor]
                chosen[s] = heavier[0] if heavier else full[s][-1]
            else:
                c = trend[s]
                chosen[s] = ActuatorChoice(**{**c.__dict__, "source": "trend-fallback",
                                              "notes": c.notes + ["目录中无满足要求的模组，用趋势模组代替"]})
        else:
            chosen[s] = trend[s]
    for s, part in fixed_parts(spec).items():
        if s in req:
            chosen[s] = check_fixed(req[s], part, spec)
    return chosen, trend, cands


# ── modules fixed by the designer ──────────────────────────────────────────────

def _norm(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


def fixed_parts(spec: dict) -> dict:
    """slot -> part entry for ``actuators.fixed`` (names looked up in the whole
    catalogue and ``extra_catalog``, whatever ``categories`` says)."""
    fixed = spec["actuators"].get("fixed") or {}
    if not fixed:
        return {}
    pool = [dict(e) for e in load_catalog()]
    extra = spec["actuators"].get("extra_catalog")
    if extra:
        pool += load_extra_catalog(extra)
    by_name = {_norm(e["name"]): e for e in pool}
    out = {}
    for slot, part in fixed.items():
        if isinstance(part, str):
            e = by_name.get(_norm(part))
            if e is None:
                close = [x["name"] for x in pool if _norm(part)[:4] in _norm(x["name"])][:6]
                raise ValueError(f"actuators.fixed.{slot}: no part named {part!r} in the "
                                 f"catalogue or extra_catalog" +
                                 (f" (similar: {', '.join(close)})" if close else ""))
            out[slot] = e
        else:
            e = {k: (float(v) if k not in ("name", "vendor", "category", "rotor_inertia_kind")
                     else v) for k, v in part.items()}
            e.setdefault("name", f"自定义-{slot}")
            e.setdefault("vendor", "custom")
            e.setdefault("category", "custom")
            if "OD_mm" in e and "radius_mm" not in e:
                e["radius_mm"] = e["OD_mm"] / 2
            out[slot] = e
    return out


def worst_task(rq: SlotRequirement, tau_peak: float, omega_nl: float, spec: dict
               ) -> tuple[str, float]:
    """The task whose samples come closest to (or go furthest past) the envelope."""
    a = spec["actuators"]
    worst, label = math.inf, ""
    for d in rq.demands:
        r = envelope_ratio(tau_peak, omega_nl, d.tau * a["torque_safety_factor"],
                           d.omega * a["speed_safety_factor"], a["voltage_limit_factor"])
        if r < worst:
            worst, label = r, d.label
    return label, worst - 1


def check_fixed(rq: SlotRequirement, part: dict, spec: dict) -> ActuatorChoice:
    """A designer-fixed module, checked against every task (never resized)."""
    tau, om, th = _scaled(rq, spec)
    c = _part_choice(part, rq.slot, tau, om, th, spec)
    c.source = "fixed"
    label, margin = worst_task(rq, part["tau_peak_Nm"], part["omega_NL_rad_s"], spec)
    c.notes = [n for n in c.notes] + [f"最紧工况：{label}（扭矩裕度 {margin:+.0%}）"]
    if c.torque_margin < 0:
        need = float(np.max(np.maximum(tau, tau / np.maximum(
            spec["actuators"]["voltage_limit_factor"] * (1 - om / part["omega_NL_rad_s"]), 1e-6))))
        c.notes.append(f"固定模组扭矩/转速不足：{label} 需要约 {need:.0f} Nm 峰值"
                       f"（在 {part['omega_NL_rad_s']:.1f} rad/s 空载转速下）")
    if c.thermal_margin < 0:
        c.notes.append(f"固定模组发热不足：持续 {th:.1f} Nm（含系数）> 持续扭矩 {c.tau_cont:.1f} Nm")
    return c
