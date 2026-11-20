"""Electrical power, battery sizing and cost of transport.

Per task, the robot's average electrical power is

    P = Σ_actuators [ P⁺_mech / η_drive  −  regen·η_drive·P⁻_mech  +  k_cu·τ² ]  +  P_compute + P_sensors

where ``k_cu`` makes an actuator dissipate ``copper_loss_W_per_kg × its mass``
at its continuous torque — a thermal-capacity argument, because the catalogue
publishes no motor constants. The mission mix weights the tasks; the battery
carries ``runtime_h`` of that mix plus a reserve.

Peak power is summed sample by sample over the actuators that work at the same
time: in one task, traces with the same phase suffix are simultaneous
(``stance``/``swing`` are the two legs at once; ``stance_up``/``swing_up`` one
stair step), and whole-body traces (``body``) add to every phase. A joint seen
only in one symmetric trace (squat, stand, arms) counts both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actuators import _motor_traces
from .planar import G


@dataclass
class TaskPower:
    task: str
    label: str
    mech_positive_W: float
    mech_negative_W: float
    copper_W: float
    electrical_W: float       # incl. compute and sensors
    peak_W: float             # sum of per-actuator peaks, incl. electronics
    per_slot_W: dict = field(default_factory=dict)


@dataclass
class EnergyBudget:
    tasks: dict               # name -> TaskPower
    mission_W: float
    battery_Wh: float
    battery_kg: float
    battery_L: float
    peak_W: float
    peak_task: str
    peak_current_A: float
    c_rate_needed: float
    c_rate_ok: bool
    cot_walk: float           # electrical cost of transport at the nominal walk
    cot_walk_actuators: float  # same, actuators only
    runtime_check_h: float    # runtime the sized pack gives on the mission mix


def _units(slot: str, n_motor_series: int, counts: dict) -> float:
    return counts.get(slot, 0) / max(n_motor_series, 1)


def _phase(tag: str) -> str:
    if tag == "body":
        return "*"
    if "_" in tag:
        return tag.split("_", 1)[1]
    return tag if tag in ("toe", "edge") else "main"


def peak_power(result, chosen: dict, counts: dict, spec: dict) -> float:
    """Largest instantaneous electrical power of the whole robot in one task."""
    e = spec["electrical"]
    eta = e["drive_efficiency"]
    parallel = spec["mechanism"]["ankle"] == "parallel"
    static = e["compute_power_W"] + e["sensor_power_W"]
    phases: dict = {}
    for slot, act in chosen.items():
        k_cu = e["copper_loss_W_per_kg"] * act.mass / max(act.tau_cont, 1e-6) ** 2
        if slot == "ankle_drive" and parallel:
            pitch = result.traces.get("ankle_pitch", [])
            roll = result.traces.get("ankle_roll", [])
            tags = [tr.tag for tr in pitch] + [tr.tag for tr in roll]
        else:
            tags = [tr.tag for tr in result.traces.get(slot, [])]
        motors = _motor_traces(result.traces, slot, parallel)
        uniq = list(dict.fromkeys(tags))
        per_leg = {"stance", "swing"} <= {t.split("_")[0] for t in uniq}
        sides = counts.get(slot, 0) / max(len(motors), 1)
        for traces in motors:
            for tag, (t, w, _) in zip(uniq, traces):
                p = np.maximum(t * w, 0) / eta + k_cu * t * t
                mult = 1.0 if (per_leg and tag.split("_")[0] in ("stance", "swing")) else sides
                ph = _phase(tag)
                phases.setdefault(ph, []).append(mult * p)
    best = 0.0
    for ph, arrs in phases.items():
        if ph == "*":
            continue
        group = arrs + phases.get("*", [])
        n = len(group[0])
        if all(len(a) == n for a in group):
            best = max(best, float(np.sum(group, axis=0).max()))
        else:
            best = max(best, float(sum(a.max() for a in group)))
    if not best and "*" in phases:
        group = phases["*"]
        n = len(group[0])
        best = (float(np.sum(group, axis=0).max()) if all(len(a) == n for a in group)
                else float(sum(a.max() for a in group)))
    return best + static


def task_power(result, chosen: dict, counts: dict, spec: dict) -> TaskPower:
    e = spec["electrical"]
    eta, regen = e["drive_efficiency"], e["regen_efficiency"]
    parallel = spec["mechanism"]["ankle"] == "parallel"
    static = e["compute_power_W"] + e["sensor_power_W"]
    pos = neg = cu = 0.0
    per_slot = {}
    for slot, act in chosen.items():
        motors = _motor_traces(result.traces, slot, parallel)
        if not any(motors):
            continue
        k_cu = e["copper_loss_W_per_kg"] * act.mass / max(act.tau_cont, 1e-6) ** 2
        units = _units(slot, len(motors), counts)
        sp = 0.0
        for traces in motors:
            if not traces:
                continue
            tot = max(sum(sh for *_, sh in traces), 1.0)
            p_pos = sum(sh * float(np.mean(np.maximum(t * w, 0))) for t, w, sh in traces) / tot
            p_neg = sum(sh * float(np.mean(np.maximum(-t * w, 0))) for t, w, sh in traces) / tot
            p_cu = sum(sh * float(np.mean(k_cu * t * t)) for t, _, sh in traces) / tot
            pos += units * p_pos
            neg += units * p_neg
            cu += units * p_cu
            sp += units * (p_pos / eta + p_cu - regen * eta * p_neg)
        per_slot[slot] = sp
    elec = pos / eta - regen * eta * neg + cu + static
    peak = peak_power(result, chosen, counts, spec)
    return TaskPower(result.name, result.label, pos, neg, cu, elec, peak, per_slot)


def energy_budget(concept, results: list, chosen: dict, spec: dict) -> EnergyBudget:
    e = spec["electrical"]
    counts = concept.mass.actuator_count
    tp = {r.name: task_power(r, chosen, counts, spec) for r in results}
    mission = sum(item["share"] * tp[item["task"]].electrical_W
                  for item in spec["mission"] if item["task"] in tp)
    usable = 1.0 - e["battery_reserve"]
    wh = mission * e["runtime_h"] / usable
    kg = wh / e["battery_Wh_per_kg"]
    # the ankle-authority check is a static capability, not a duty: no peak from it
    peak_task = max((x for k, x in tp.items() if k != "balance"), key=lambda x: x.peak_W)
    c_needed = peak_task.peak_W / max(wh, 1e-9)
    M = concept.mass.total
    walk = tp.get("walk")
    v = spec["tasks"]["walk"]["speed"]
    if walk is not None:
        cot = walk.electrical_W / (M * G * v)
        static = e["compute_power_W"] + e["sensor_power_W"]
        cot_a = (walk.electrical_W - static) / (M * G * v)
    else:
        cot = cot_a = float("nan")
    return EnergyBudget(
        tasks=tp, mission_W=mission, battery_Wh=wh, battery_kg=kg,
        battery_L=wh / e["battery_Wh_per_L"], peak_W=peak_task.peak_W,
        peak_task=peak_task.label, peak_current_A=peak_task.peak_W / e["bus_voltage_V"],
        c_rate_needed=c_needed, c_rate_ok=c_needed <= e["battery_max_C"],
        cot_walk=cot, cot_walk_actuators=cot_a,
        runtime_check_h=wh * usable / max(mission, 1e-9))
