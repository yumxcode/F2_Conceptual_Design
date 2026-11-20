"""The mass loop: a concept's actuators, battery and total mass, solved together.

Heavier actuators make a heavier robot, which needs stronger actuators and a
bigger battery. :func:`evaluate` iterates

    masses -> tasks -> requirements -> actuators + battery -> masses

until the total mass stops moving. A loop that keeps growing past
``convergence.max_mass_kg`` is a concept the hardware cannot close (the
"mass spiral"), and is reported as such rather than as a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..trends import link_fits
from .actuators import choose, requirements
from .body import Concept, actuator_slots, build_concept, geometry
from .energy import EnergyBudget, energy_budget
from .spec import dof_count
from .tasks import run_tasks


@dataclass
class ConceptResult:
    spec: dict
    concept: Concept
    tasks: list
    req: dict
    chosen: dict
    trend: dict
    candidates: dict
    energy: EnergyBudget
    converged: bool
    closed: bool              # False: mass spiral or an actuator beyond any part
    iterations: int
    history: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def mass(self) -> float:
        return self.concept.mass.total

    def warnings(self) -> list[dict]:
        out = []
        for r in self.tasks:
            for c in r.checks:
                if not c["ok"]:
                    out.append({**c, "task": r.label})
        for s, a in self.chosen.items():
            if not a.feasible:
                out.append({"name": f"{s}: 执行器不可行", "note": "; ".join(a.notes),
                            "severity": "error", "task": "-", "ok": False})
        if not self.energy.c_rate_ok:
            out.append({"name": "电池放电倍率", "severity": "warn", "task": self.energy.peak_task,
                        "ok": False,
                        "note": f"峰值功率需要 {self.energy.c_rate_needed:.1f}C，"
                                f"超过设定 {self.spec['electrical']['battery_max_C']:g}C"})
        return out

    def joint_peaks(self) -> dict:
        """Joint-level peak |torque| and |speed| over all tasks (before any
        parallel-linkage split), with the task that set each."""
        out: dict = {}
        for r in self.tasks:
            for j, trs in r.traces.items():
                for tr in trs:
                    t = float(abs(tr.tau).max()) if len(tr.tau) else 0.0
                    w = float(abs(tr.omega).max()) if len(tr.omega) else 0.0
                    d = out.setdefault(j, {"tau": 0.0, "omega": 0.0, "tau_task": "", "omega_task": ""})
                    if t > d["tau"]:
                        d["tau"], d["tau_task"] = t, r.label
                    if w > d["omega"]:
                        d["omega"], d["omega_task"] = w, r.label
        return out

    def summary(self) -> dict:
        """One flat row (sweeps, CSV)."""
        g = self.concept.geo
        b = self.concept.mass.breakdown()
        row = {
            "name": self.spec["name"],
            "height_m": round(g.height, 3),
            "leg_ratio": self.spec["body"]["leg_ratio"],
            "leg_m": round(g.leg, 3),
            "thigh_m": round(g.thigh, 3), "shank_m": round(g.shank, 3),
            "dof": dof_count(self.spec),
            "mass_kg": round(self.mass, 2) if self.closed else math.inf,
            "actuator_kg": round(b["actuators"], 2),
            "structure_kg": round(b["limb_structure"], 2),
            "battery_kg": round(b["battery"], 2),
            "battery_Wh": round(self.energy.battery_Wh, 0),
            "mission_W": round(self.energy.mission_W, 0),
            "peak_W": round(self.energy.peak_W, 0),
            "cot_walk": round(self.energy.cot_walk, 3),
            "converged": self.converged, "closed": self.closed,
            "n_warnings": len(self.warnings()),
            "n_infeasible_actuators": sum(1 for a in self.chosen.values() if not a.feasible),
        }
        for s, rq in self.req.items():
            row[f"{s}_tau_Nm"] = round(rq.peak_tau, 1)
            row[f"{s}_omega_rad_s"] = round(rq.peak_omega, 1)
            row[f"{s}_rms_Nm"] = round(rq.thermal_rms, 1)
            row[f"{s}_part"] = self.chosen[s].name
            row[f"{s}_part_kg"] = round(self.chosen[s].mass, 3)
        return row


def initial_guess(spec: dict) -> tuple[dict, float]:
    """Actuator masses and battery from Draft's whole-robot trends."""
    geo = geometry(spec)
    lt = link_fits("humanoid")
    mv = lt.mass_vs_height
    M0 = mv["coef"] * geo.z_span ** mv["exp"]
    slots = actuator_slots(spec)
    share = lt.actuator_fraction["median"] * M0
    n = sum(4 if s == "ankle_drive" else (1 if s.startswith("waist") else 2) for s in slots)
    return {s: share / n for s in slots}, 0.06 * M0


def _loop(spec: dict, act_mass: dict, armature: dict, battery: float,
          monotone: bool) -> ConceptResult:
    conv = spec["convergence"]
    slots = actuator_slots(spec)
    history, notes = [], []
    converged, closed = False, True
    prev_M = None
    result = None
    for it in range(1, conv["max_iter"] + 1):
        concept = build_concept(spec, act_mass, armature, battery)
        tasks = run_tasks(concept)
        req = requirements(tasks, slots, spec)
        sticky = monotone and it > 1
        chosen, trend, cands = choose(req, spec, act_mass if sticky else None)
        energy = energy_budget(concept, tasks, chosen, spec)
        M = concept.mass.total
        history.append({"iter": it, "mass_kg": M, "battery_kg": battery,
                        "actuator_kg": concept.mass.breakdown()["actuators"]})
        result = ConceptResult(spec, concept, tasks, req, chosen, trend, cands, energy,
                               False, True, it, history, notes)
        new_mass = {s: chosen[s].mass for s in slots}
        if any(not math.isfinite(m) for m in new_mass.values()):
            closed = False
            notes.append("某个关节所需转速超出任何减速比，无法闭合")
            break
        if M > conv["max_mass_kg"]:
            closed = False
            notes.append(f"质量螺旋：总质量超过 {conv['max_mass_kg']:g} kg 仍在增长")
            break
        if sticky:
            # discrete parts can flip between two choices; only let masses grow
            new_mass = {s: max(new_mass[s], act_mass[s]) for s in slots}
        if prev_M is not None and abs(M - prev_M) / prev_M < conv["tol"] and \
                all(abs(new_mass[s] - act_mass[s]) <= 0.02 * max(act_mass[s], 1e-3)
                    for s in slots):
            converged = True
            break
        # the reduction grid is discrete, so a loop can settle into a narrow band
        recent = [h["mass_kg"] for h in history[-6:]]
        if len(recent) == 6 and (max(recent) - min(recent)) / max(recent) < 0.01:
            converged = True
            notes.append(f"质量在 {min(recent):.1f}–{max(recent):.1f} kg 之间小幅摆动（<1%），视为收敛")
            break
        if not sticky and it > 10:
            # a slow or two-cycle loop (a choice flipping at the density frontier):
            # under-relax so it settles
            new_mass = {s: 0.5 * (new_mass[s] + act_mass[s]) for s in slots}
        act_mass = new_mass
        armature = {s: chosen[s].armature for s in slots}
        battery = max(battery, energy.battery_kg) if sticky else energy.battery_kg
        prev_M = M
    result.converged, result.closed = converged, closed
    result.iterations = len(history)
    if not converged and closed:
        notes.append("质量迭代未在限定次数内收敛")
    return result


def evaluate(spec: dict) -> ConceptResult:
    """Close the mass loop. Catalogue mode starts from the converged trend design
    and only lets actuator masses grow, so a discrete choice cannot oscillate."""
    act_mass, battery = initial_guess(spec)
    slots = actuator_slots(spec)
    armature = {s: 0.0 for s in slots}
    if spec["actuators"]["mode"] == "trend":
        return _loop(spec, act_mass, armature, battery, monotone=False)
    trend_spec = {**spec, "actuators": {**spec["actuators"], "mode": "trend"}}
    first = _loop(trend_spec, act_mass, armature, battery, monotone=False)
    if not first.closed:
        return first
    start = {s: first.chosen[s].mass for s in slots}
    arm = {s: first.chosen[s].armature for s in slots}
    # start a little light so the catalogue choice, not the trend, sets the mass
    start = {s: 0.8 * m for s, m in start.items()}
    return _loop(spec, start, arm, 0.8 * first.energy.battery_kg, monotone=True)
