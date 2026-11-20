"""The duty cycles a concept is sized against.

Each task prescribes a whole-body motion (pelvis path, foot paths, trunk and
arm angles), solves the leg inverse kinematics, and runs planar inverse
dynamics. What comes out is, for every joint, one or more time traces of
torque and speed *per actuator*, each with the share of the task cycle it
represents (a leg is in stance half the stride and in swing the other half).

Motions are deliberately simple and scale with the body:

* ``stand``      static double support, knees slightly bent (optional payload)
* ``walk``       single support, linear-inverted-pendulum CoM, Froude-scaled steps;
                 sagittal and frontal (hip/ankle roll) models on one time grid
* ``stairs``     one step up and one step down, CoM raised/lowered by the riser
* ``squat_lift`` symmetric squat down and up with the payload held in front,
                 trunk lean solved so the CoM stays over the feet
* ``stoop_lift`` bend forward at the hips to the payload and back (hip-dominant)
* ``run``        spring-mass stance: sinusoidal vertical force from the duty factor,
                 horizontal force along the leg, swing leg on a Froude-scaled cycle
* ``arm_lift``   both arms raised straight from hanging to horizontal with payload
* ``turn``       turning in place (hip yaw against foot friction, waist yaw inertia)
* ``balance``    quasi-static ankle authority: CoP at the toe and at the foot edge

None of these is a controller. They are reference motions a capable controller
would produce, used to bound what the hardware must deliver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .body import Concept, stand_hip_height
from .planar import G, PlanarBody, PlanarTree, angle_series, derivatives, static_com

N_SAMPLES = 241
UP = math.pi / 2
DOWN = -math.pi / 2


@dataclass
class Trace:
    """Torque and speed of ONE actuator over part of a task cycle."""
    tag: str                  # pairs simultaneous traces (stance/swing/static)
    t: np.ndarray
    tau: np.ndarray
    omega: np.ndarray
    share: float              # fraction of the task cycle this trace covers


@dataclass
class TaskResult:
    name: str
    label: str
    continuous: bool          # sized for thermal (RMS) duty as well as peak
    traces: dict              # joint -> [Trace]
    checks: list = field(default_factory=list)   # dicts: name, value, limit, ok, note
    info: dict = field(default_factory=dict)

    def add(self, joint: str, trace: Trace) -> None:
        self.traces.setdefault(joint, []).append(trace)


# ── kinematics helpers ─────────────────────────────────────────────────────────

def minjerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def bump(s: np.ndarray) -> np.ndarray:
    """0 -> 1 -> 0 with zero slope at both ends."""
    return 0.5 * (1 - np.cos(2 * math.pi * np.clip(s, 0, 1)))


def leg_ik(hip: np.ndarray, ankle: np.ndarray, l1: float, l2: float
           ) -> tuple[np.ndarray, float]:
    """Knee position for a two-link leg, knee forward. Returns (knee, max d/(l1+l2))."""
    v = ankle - hip
    d = np.linalg.norm(v, axis=1)
    reach = d / (l1 + l2)
    d = np.clip(d, abs(l1 - l2) + 1e-6, (l1 + l2) * 0.9995)
    psi = np.arctan2(v[:, 1], v[:, 0])
    beta = np.arccos(np.clip((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d), -1, 1))
    th = psi + beta
    knee = hip + l1 * np.stack([np.cos(th), np.sin(th)], axis=1)
    return knee, float(reach.max())


def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    return np.arctan2(v[:, 1], v[:, 0])


def _const(T: int, value: float) -> np.ndarray:
    return np.full(T, float(value))


# ── running a sagittal motion ─────────────────────────────────────────────────

def _sagittal_angles(concept: Concept, hip, trunk, swing, arm, double_support,
                     foot_pitch_swing):
    """Absolute body angles of the sagittal tree for a hip path and a swing spec.

    ``swing`` is ``("ankle", (T,2))`` for an ankle path solved by IK, or
    ``("angles", (thigh, shank))`` for absolute swing-leg angles.
    """
    g = concept.geo
    T = len(hip)
    zero = np.zeros((T, 2))
    knee, reach = leg_ik(hip, zero, g.thigh, g.shank)
    ang = {
        "st_shank": _ang(zero, knee),
        "st_thigh": _ang(knee, hip),
        "pelvis": np.asarray(trunk, float) * np.ones(T),
        "torso": np.asarray(trunk, float) * np.ones(T),
        "upper_arm": np.asarray(arm[0], float) * np.ones(T),
        "forearm": np.asarray(arm[1], float) * np.ones(T),
    }
    if not double_support:
        kind, val = swing
        if kind == "angles":
            ang["sw_thigh"], ang["sw_shank"] = val
        else:
            sk, sr = leg_ik(hip, val, g.thigh, g.shank)
            reach = max(reach, sr)
            ang["sw_thigh"] = _ang(hip, sk)
            ang["sw_shank"] = _ang(sk, val)
        ang["sw_foot"] = (foot_pitch_swing if foot_pitch_swing is not None else np.zeros(T))
    return ang, reach


def _run_sagittal(concept: Concept, t, hip, trunk, *, swing=None, arm=(DOWN, DOWN),
                  double_support=False, payload=0.0, foot_pitch_swing=None):
    tree = concept.sagittal(double_support=double_support, payload=payload)
    ang, reach = _sagittal_angles(concept, hip, trunk, swing, arm, double_support,
                                  foot_pitch_swing)
    phi, dphi, ddphi = angle_series(t, ang)
    return tree, tree.inverse_dynamics(phi, dphi, ddphi), reach


def _plan_hip(concept: Concept, com_target, trunk, swing_fn, *, arm=(DOWN, DOWN),
              double_support=False, payload=0.0, iters=6):
    """Hip path whose whole-body CoM follows ``com_target`` (T,2).

    The swing leg and arms move the CoM relative to the hip; a few fixed-point
    passes put the CoM, not the hip, on the planned path.
    """
    tree = concept.sagittal(double_support=double_support, payload=payload)
    hip = com_target - np.array([0.0, _upright_com_offset(concept, payload)])
    swing = None
    for _ in range(iters):
        swing = swing_fn(hip) if swing_fn else None
        ang, _ = _sagittal_angles(concept, hip, trunk, swing, arm, double_support, None)
        phi = {k: np.asarray(v, float) for k, v in ang.items()}
        z = {k: np.zeros_like(v) for k, v in phi.items()}
        err = com_target - tree.inverse_dynamics(phi, z, z).com_world
        hip = hip + err
        if np.abs(err).max() < 2e-4:
            break
    return hip, (swing_fn(hip) if swing_fn else None)


def pendulum_x(t, zc, x0, x1, pivot_z, pivot_x=0.0):
    """CoM path of an inverted pendulum on a fixed CoP with prescribed height.

    Solves ``x'' = (z''+g)/(z - pivot_z) · (x - pivot_x)`` with ``x(0)=x0,
    x(T)=x1``; for a constant height this is the linear inverted pendulum.
    """
    x0, x1 = x0 - pivot_x, x1 - pivot_x
    _, zdd = derivatives(t, zc)
    k = (zdd + G) / np.maximum(zc - pivot_z, 1e-3)

    def rhs(tt, y):
        kk = np.interp(tt, t, k)
        return [y[1], kk * y[0], y[3], kk * y[2]]
    sol = solve_ivp(rhs, (t[0], t[-1]), [1.0, 0.0, 0.0, 1.0], t_eval=t,
                    rtol=1e-9, atol=1e-12)
    u, w = sol.y[0], sol.y[2]
    c = (x1 - x0 * u[-1]) / w[-1]
    return pivot_x + x0 * u + c * w


def _sole_centre(g) -> float:
    """Middle of the sole, relative to the ankle."""
    return 0.5 * g.foot_length - g.heel


def _swing_path(t, hip, start, end, clear):
    """Swing ankle: interpolated relative to the hip, exact at both ends."""
    s = minjerk((t - t[0]) / (t[-1] - t[0]))
    r0 = np.asarray(start) - hip[0]
    r1 = np.asarray(end) - hip[-1]
    rel = r0 + (r1 - r0) * s[:, None]
    rel[:, 1] += clear * np.sin(math.pi * (t - t[0]) / (t[-1] - t[0])) ** 2
    return hip + rel


def _cop_check(concept: Concept, res, label: str, strict: bool = True) -> dict:
    """Centre of pressure of the stance foot against its sole (loaded samples)."""
    g = concept.geo
    Fh, Fz = res.ground_force[:, 0], res.ground_force[:, 1]
    tau_a = res.tau["ankle_pitch"]
    loaded = Fz > 0.3 * Fz.max()
    # ground moment about the ankle equals the ankle torque (massless foot);
    # CoP at (x, -ankle_h): x·Fz + ankle_h·Fh = tau_a
    x = (tau_a[loaded] - g.ankle_h * Fh[loaded]) / Fz[loaded]
    lo, hi = -g.heel, g.toe
    worst = float(max(x.max() - hi, lo - x.min(), 0.0))
    # A warning, not a refusal: the planar reference motions use no trunk or arm
    # momentum to help balance, so a small excursion means "foot is marginal".
    # Fast gaits are informational: heel-strike and toe-off use the sole's edges
    # and the swing leg's momentum is real, so only nominal motions are flagged.
    return {"name": f"{label}: CoP inside the sole", "value": round(worst * 1000, 1),
            "limit": 0.0, "unit": "mm outside", "ok": worst <= 0.005 or not strict,
            "severity": "warn" if strict else "info",
            "note": f"CoP {x.min()*1000:.0f}..{x.max()*1000:.0f} mm, "
                    f"sole {lo*1000:.0f}..{hi*1000:.0f} mm (ankle at 0)"}


def _sagittal_traces(result: TaskResult, t, res, *, tag_stance="stance",
                     tag_swing="swing", share_stance=0.5, share_swing=0.5,
                     double_support=False):
    """Copy inverse-dynamics output into per-actuator traces."""
    arm_div = 1.0 if double_support else 2.0
    waist_mul = 2.0 if double_support else 1.0
    m = {"ankle_pitch": ("ankle_pitch", tag_stance, share_stance, 1.0),
         "knee": ("knee", tag_stance, share_stance, 1.0),
         "hip_pitch": ("hip_pitch", tag_stance, share_stance, 1.0),
         "waist_pitch": ("waist_pitch", "body", 1.0, waist_mul),
         "shoulder_pitch": ("shoulder_pitch", "body", 1.0, 1.0 / arm_div),
         "elbow": ("elbow", "body", 1.0, 1.0 / arm_div),
         "hip_pitch_swing": ("hip_pitch", tag_swing, share_swing, 1.0),
         "knee_swing": ("knee", tag_swing, share_swing, 1.0),
         "ankle_pitch_swing": ("ankle_pitch", tag_swing, share_swing, 1.0)}
    for jn, tau in res.tau.items():
        joint, tag, share, mul = m[jn]
        result.add(joint, Trace(tag, t, tau * mul, res.qd[jn], share))


def _upright_com_offset(concept: Concept, payload: float = 0.0) -> float:
    """Height of the whole-body CoM above the hip in the upright pose."""
    g = concept.geo
    tree = concept.sagittal(double_support=False, payload=payload)
    ang = {"st_shank": UP, "st_thigh": UP, "pelvis": UP, "torso": UP,
           "upper_arm": DOWN, "forearm": DOWN, "sw_thigh": DOWN, "sw_shank": DOWN,
           "sw_foot": 0.0}
    com = static_com(tree, ang)
    return float(com[1] - g.leg)


def _reach_check(label: str, reach: float) -> dict:
    return {"name": f"{label}: leg reach", "value": round(reach, 3), "limit": 0.999,
            "unit": "of leg length", "ok": reach <= 0.999,
            "note": "the motion asks for more than a straight leg" if reach > 0.999 else ""}


# ── tasks ──────────────────────────────────────────────────────────────────────

def task_stand(concept: Concept, p: dict) -> TaskResult:
    g = concept.geo
    hold = bool(p.get("hold_payload", False))
    payload = concept.mass.payload_hands if hold else 0.0
    h = stand_hip_height(g, p["knee_bend_deg"]) - g.ankle_h
    arm = (DOWN, 0.0) if hold else (DOWN, DOWN)
    tree = concept.sagittal(double_support=True, payload=payload)
    target = 0.5 * g.foot_length - g.heel

    def com_x(xh):
        hp = np.array([[xh, h]])
        ang, _ = _sagittal_angles(concept, hp, UP, None, arm, True, None)
        return static_com(tree, {k: float(np.asarray(v).ravel()[0]) for k, v in ang.items()})[0] - target
    lim = 0.4 * g.leg
    xh = brentq(com_x, -lim, lim) if com_x(-lim) * com_x(lim) < 0 else 0.0
    t = np.linspace(0, p["duration_s"], 21)
    hip = np.tile([xh, h], (len(t), 1))
    _, res, _ = _run_sagittal(concept, t, hip, UP, arm=arm, double_support=True,
                              payload=payload)
    out = TaskResult("stand", "站立" + ("（双手持重）" if hold else ""), True, {},
                     info={"hip_height_m": h + g.ankle_h})
    _sagittal_traces(out, t, res, tag_stance="stand", share_stance=1.0, double_support=True)
    out.checks.append(_cop_check(concept, res, "stand"))
    _static_upper_limbs(concept, out, t, holding=hold)
    return out


def _walk_geometry(concept: Concept, p: dict, v: float):
    g = concept.geo
    leg = g.leg
    fr = v * v / (G * leg)
    if p.get("step_length_ratio"):
        Ls = p["step_length_ratio"] * leg
    else:
        Ls = leg * 0.8 * (max(fr, 1e-3) / 0.2) ** 0.35
    ext = p["max_leg_extension"] * leg
    Ls = min(Ls, 0.96 * ext)
    h = math.sqrt(max(ext ** 2 - (Ls / 2) ** 2, (0.6 * leg) ** 2))
    return Ls, h, fr


def task_walk(concept: Concept, p: dict, speed: Optional[float] = None,
              name: str = "walk") -> TaskResult:
    g = concept.geo
    v = float(speed if speed is not None else p["speed"])
    Ls, h, fr = _walk_geometry(concept, p, v)
    Tss = Ls / v
    zc = h + _upright_com_offset(concept)
    t = np.linspace(0, Tss, N_SAMPLES)
    xc = pendulum_x(t, np.full_like(t, zc), -Ls / 2, Ls / 2, -g.ankle_h, _sole_centre(g))
    com = np.stack([xc, np.full_like(t, zc)], axis=1)
    clear = p["swing_height_ratio"] * g.height

    def swing(hip):
        return ("ankle", _swing_path(t, hip, (-Ls, 0.0), (Ls, 0.0), clear))
    hip, sw = _plan_hip(concept, com, UP, swing)
    _, res, reach = _run_sagittal(concept, t, hip, UP, swing=sw)
    label = f"步行 {v:.2f} m/s"
    out = TaskResult(name, label, name == "walk", {},
                     info={"speed_m_s": v, "step_length_m": Ls, "step_time_s": Tss,
                           "cadence_steps_min": 60 / Tss,
                           "hip_height_m": float(hip[:, 1].mean()) + g.ankle_h,
                           "froude": fr, "leg_reach": reach})
    _sagittal_traces(out, t, res)
    out.checks.append(_cop_check(concept, res, label, strict=(name == "walk")))
    out.checks.append(_reach_check(label, reach))
    Tc = math.sqrt((zc + g.ankle_h) / G)
    _frontal_walk(concept, out, t, float(hip[:, 1].mean()), Tss, Tc, p, clear)
    _static_upper_limbs(concept, out, t, holding=False)
    return out


def _frontal_walk(concept: Concept, out: TaskResult, t, h, Tss, Tc, p, clear):
    """Hip and ankle roll during single support (frontal LIPM sway)."""
    g = concept.geo
    w = g.hip_width
    wstep = (p.get("step_width_ratio") or 0.0) * g.height or w
    tree: PlanarTree = concept.frontal(h + g.ankle_h, max(h - clear / 2, 0.5 * g.leg))
    y = (wstep / 2) * np.cosh((t - Tss / 2) / Tc) / math.cosh(Tss / (2 * Tc))
    st_hip = np.stack([y - w / 2, np.full_like(t, h)], axis=1)
    sw_hip = st_hip + np.array([w, 0.0])
    s = t / Tss
    sw_ank = np.stack([np.full_like(t, wstep), clear * np.sin(math.pi * s) ** 2], axis=1)
    ang = {"st_leg": _ang(np.zeros_like(st_hip), st_hip), "pelvis": np.zeros_like(t),
           "sw_leg": _ang(sw_hip, sw_ank)}
    if "torso" in tree.index:
        ang["torso"] = np.full_like(t, UP)
    phi, dphi, ddphi = angle_series(t, ang)
    res = tree.inverse_dynamics(phi, dphi, ddphi)
    out.add("ankle_roll", Trace("stance", t, res.tau["ankle_roll"], res.qd["ankle_roll"], 0.5))
    out.add("hip_roll", Trace("stance", t, res.tau["hip_roll"], res.qd["hip_roll"], 0.5))
    out.add("hip_roll", Trace("swing", t, res.tau["hip_roll_swing"], res.qd["hip_roll_swing"], 0.5))
    out.add("ankle_roll", Trace("swing", t, np.zeros_like(t), np.zeros_like(t), 0.5))
    if "waist_roll" in res.tau:
        out.add("waist_roll", Trace("body", t, res.tau["waist_roll"], res.qd["waist_roll"], 1.0))
    out.info["lateral_sway_m"] = float(y.max() - y.min())


def task_stairs(concept: Concept, p: dict) -> TaskResult:
    g = concept.geo
    Hs, tread = p["step_height_m"], p["tread_m"]
    T = p["step_time_s"] or 2.4 * math.sqrt(g.leg / G)
    ext = concept.spec["tasks"]["walk"]["max_leg_extension"] * g.leg
    h = math.sqrt(max(ext ** 2 - (tread / 2) ** 2, (0.6 * g.leg) ** 2))
    d = _upright_com_offset(concept)
    t = np.linspace(0, T, N_SAMPLES)
    s = minjerk(t / T)
    clear = 0.04 * g.height
    out = TaskResult("stairs", f"楼梯 {Hs*100:.0f} cm 上/下", False, {},
                     info={"step_height_m": Hs, "tread_m": tread, "step_time_s": T})
    worst_reach = 0.0
    for direction, sign in (("up", 1.0), ("down", -1.0)):
        z0, z1 = (h - Hs, h) if sign > 0 else (h, h - Hs)
        zc = d + z0 + (z1 - z0) * s
        xc = pendulum_x(t, zc, -tread / 2, tread / 2, -g.ankle_h, _sole_centre(g))
        com = np.stack([xc, zc], axis=1)

        def swing(hip, sign=sign):
            return ("ankle", _swing_path(t, hip, (-tread, -sign * Hs), (tread, sign * Hs),
                                         clear + max(sign, 0) * Hs * 0.5))
        hip, sw = _plan_hip(concept, com, UP, swing)
        _, res, reach = _run_sagittal(concept, t, hip, UP, swing=sw)
        worst_reach = max(worst_reach, reach)
        _sagittal_traces(out, t, res, tag_stance=f"stance_{direction}",
                         tag_swing=f"swing_{direction}", share_stance=0.25, share_swing=0.25)
        out.checks.append(_cop_check(concept, res, f"stairs {direction}"))
    out.checks.append(_reach_check("stairs", worst_reach))
    return out


def task_squat_lift(concept: Concept, p: dict) -> TaskResult:
    g = concept.geo
    payload = concept.mass.payload_hands if p["carry_payload"] else 0.0
    h_top = stand_hip_height(g, 20.0) - g.ankle_h
    h_low = p["depth"] * g.leg
    D = p["duration_s"]
    t = np.linspace(0, D, N_SAMPLES)
    # down and back up, each half a minimum-jerk move (rest at top and bottom)
    s = t / D
    prof = np.where(s < 0.5, minjerk(2 * s), minjerk(2 - 2 * s))
    zh = h_top - (h_top - h_low) * prof
    xh = -0.45 * (h_top - zh)
    hip = np.stack([xh, zh], axis=1)
    tree = concept.sagittal(double_support=True, payload=payload)
    target = 0.5 * g.foot_length - g.heel
    knee, _ = leg_ik(hip, np.zeros_like(hip), g.thigh, g.shank)
    a_sh, a_th = _ang(np.zeros_like(hip), knee), _ang(knee, hip)
    lean = np.empty_like(t)
    lo, hi = UP - 1.3, UP + 0.4
    for i in range(len(t)):
        def f(th, i=i):
            return static_com(tree, {"st_shank": a_sh[i], "st_thigh": a_th[i], "pelvis": th,
                                     "torso": th, "upper_arm": DOWN, "forearm": 0.0})[0] - target
        flo, fhi = f(lo), f(hi)
        lean[i] = brentq(f, lo, hi) if flo * fhi < 0 else (lo if abs(flo) < abs(fhi) else hi)
    _, res, reach = _run_sagittal(concept, t, hip, lean, arm=(DOWN, 0.0),
                                  double_support=True, payload=payload)
    out = TaskResult("squat_lift", f"深蹲搬运 {payload:.0f} kg", False, {},
                     info={"lowest_hip_m": h_low + g.ankle_h, "max_trunk_lean_deg":
                           float(np.degrees(UP - lean.min())), "payload_kg": payload})
    _sagittal_traces(out, t, res, tag_stance="squat", share_stance=1.0, double_support=True)
    out.checks.append(_cop_check(concept, res, "squat_lift"))
    out.checks.append(_reach_check("squat_lift", reach))
    return out


def task_stoop_lift(concept: Concept, p: dict) -> TaskResult:
    """Bend forward at the hips (legs nearly straight) to the payload and back up.

    The hip-dominant lift: the trunk pitches down by ``lean_deg`` and returns;
    the pelvis slides back so the CoM stays over the feet.
    """
    g = concept.geo
    payload = concept.mass.payload_hands if p["carry_payload"] else 0.0
    h = stand_hip_height(g, p["knee_bend_deg"]) - g.ankle_h
    D = p["duration_s"]
    t = np.linspace(0, D, N_SAMPLES)
    s = t / D
    prof = np.where(s < 0.5, minjerk(2 * s), minjerk(2 - 2 * s))
    lean = UP - math.radians(p["lean_deg"]) * prof
    tree = concept.sagittal(double_support=True, payload=payload)
    target = _sole_centre(g)
    arm = (DOWN, DOWN)
    xh = np.empty_like(t)
    lim = 0.7 * h
    # the hip slides back on a circle about the ankle (constant knee bend)
    for i in range(len(t)):
        def f(x, i=i):
            hp = np.array([[x, math.sqrt(h * h - x * x)]])
            ang, _ = _sagittal_angles(concept, hp, lean[i], None, arm, True, None)
            return static_com(tree, {k: float(np.asarray(v).ravel()[0])
                                     for k, v in ang.items()})[0] - target
        flo, fhi = f(-lim), f(lim)
        xh[i] = brentq(f, -lim, lim) if flo * fhi < 0 else (-lim if abs(flo) < abs(fhi) else lim)
    hip = np.stack([xh, np.sqrt(h * h - xh * xh)], axis=1)
    _, res, reach = _run_sagittal(concept, t, hip, lean, arm=arm, double_support=True,
                                  payload=payload)
    out = TaskResult("stoop_lift", f"弯腰搬运 {payload:.0f} kg", False, {},
                     info={"lean_deg": p["lean_deg"], "hip_shift_m": float(-xh.min()),
                           "payload_kg": payload})
    _sagittal_traces(out, t, res, tag_stance="stoop", share_stance=1.0, double_support=True)
    out.checks.append(_cop_check(concept, res, "stoop_lift"))
    out.checks.append(_reach_check("stoop_lift", reach))
    return out


def task_run(concept: Concept, p: dict) -> TaskResult:
    g = concept.geo
    leg = g.leg
    v = float(p["speed"])
    f_step = 0.85 * math.sqrt(G / leg) * (1 + 0.05 * (v - 2.5))
    T_step = 1.0 / f_step
    tc = 2 * p["duty_factor"] * T_step
    M = concept.mass.total
    Fp = math.pi * M * G * T_step / (2 * tc)
    d_com = _upright_com_offset(concept)
    # vertical: sinusoidal force, symmetric about mid-stance
    k = Fp * tc / (math.pi * M)
    v0 = (2 * k - G * tc) / 2
    t = np.linspace(0, tc, N_SAMPLES)
    z_rel = -v0 * t + k * (t - (tc / math.pi) * np.sin(math.pi * t / tc)) - G * t ** 2 / 2
    x_e = v * tc / 2
    ext = p["touchdown_extension"] * leg
    z_td = math.sqrt(max(ext ** 2 - x_e ** 2, (0.5 * leg) ** 2)) + d_com
    zc = z_td + z_rel
    xc = pendulum_x(t, zc, -x_e, x_e, -g.ankle_h, _sole_centre(g))
    com = np.stack([xc, zc], axis=1)
    T_str = 2 * T_step
    A = min(0.25 + 0.12 * v, 0.8)
    kmax = min(0.6 + 0.3 * v, 2.0)
    ph = 2 * math.pi * (t - tc / 2) / T_str
    thigh_abs = DOWN + A * np.sin(ph)
    knee_flex = 0.3 + (kmax - 0.3) * np.cos(ph / 2) ** 2
    swing_angles = ("angles", (thigh_abs, thigh_abs - knee_flex))
    hip, sw = _plan_hip(concept, com, UP, lambda hp: swing_angles)
    _, res, reach = _run_sagittal(concept, t, hip, UP, swing=sw,
                                  foot_pitch_swing=np.zeros_like(t))
    out = TaskResult("run", f"跑步 {v:.1f} m/s", False, {},
                     info={"speed_m_s": v, "contact_time_s": tc, "step_time_s": T_step,
                           "peak_grf_bw": Fp / (M * G), "leg_reach": reach})
    _sagittal_traces(out, t, res, share_stance=p["duty_factor"],
                     share_swing=1 - p["duty_factor"])
    out.checks.append(_cop_check(concept, res, out.label, strict=False))
    out.checks.append(_reach_check(out.label, reach))
    # frontal: support the upper body over one leg with a vertical load of Fp
    w = g.hip_width
    tree = concept.frontal(float(hip[:, 1].mean()) + g.ankle_h, 0.8 * leg)
    # running steps are narrow: the pelvis centre sits nearly over the stance foot
    st_hip = np.stack([np.full_like(t, 0.1 * w - w / 2), hip[:, 1]], axis=1)
    sw_hip = st_hip + np.array([w, 0.0])
    sw_ank = sw_hip + np.array([0.0, -0.8 * leg])
    ang = {"st_leg": _ang(np.zeros_like(st_hip), st_hip), "pelvis": np.zeros_like(t),
           "sw_leg": _ang(sw_hip, sw_ank)}
    if "torso" in tree.index:
        ang["torso"] = np.full_like(t, UP)
    phi = {k_: np.asarray(v_, float) for k_, v_ in ang.items()}
    z = {k_: np.zeros_like(v_) for k_, v_ in phi.items()}
    rf = tree.inverse_dynamics(phi, z, z)
    load = (Fp * np.sin(math.pi * t / tc)) / (M * G)
    zero = np.zeros_like(t)
    out.add("hip_roll", Trace("stance", t, rf.tau["hip_roll"] * load, zero, p["duty_factor"]))
    out.add("ankle_roll", Trace("stance", t, rf.tau["ankle_roll"] * load, zero, p["duty_factor"]))
    if "waist_roll" in rf.tau:
        out.add("waist_roll", Trace("body", t, rf.tau["waist_roll"] * load, zero, 1.0))
    return out


def task_arm_lift(concept: Concept, p: dict) -> TaskResult:
    g = concept.geo
    D = p["duration_s"]
    t = np.linspace(0, D, N_SAMPLES)
    th = DOWN + (UP) * bump(t / D)                 # down -> horizontal -> down
    m_arm, _ = concept.arm_mass()
    # one arm, rooted at the shoulder (the trunk holds still)
    upper, fore = concept._arm_parts(0.5, concept.mass.payload_hands)
    um, uc, ui = concept._rigid(upper)
    fm, fc, fi = concept._rigid(fore)
    tree = PlanarTree([
        PlanarBody("upper_arm", -1, (0.0, 0.0), "shoulder_pitch", um, uc, ui,
                   concept.mass.armature.get("shoulder_pitch", 0.0)),
        PlanarBody("forearm", 0, (g.upper_arm, 0.0), "elbow", fm, fc, fi,
                   concept.mass.armature.get("elbow", 0.0)),
    ])
    phi, dphi, ddphi = angle_series(t, {"upper_arm": th, "forearm": th})
    res = tree.inverse_dynamics(phi, dphi, ddphi)
    out = TaskResult("arm_lift", f"双臂平举 {concept.mass.payload_hands:.0f} kg", False, {},
                     info={"arm_mass_kg": m_arm, "duration_s": D})
    sp = res.tau["shoulder_pitch"]
    out.add("shoulder_pitch", Trace("body", t, sp, res.qd["shoulder_pitch"], 1.0))
    out.add("elbow", Trace("body", t, res.tau["elbow"], res.qd["elbow"], 1.0))
    # the same raise sideways loads shoulder roll the same way
    out.add("shoulder_roll", Trace("body", t, sp, res.qd["shoulder_pitch"], 1.0))
    return out


def _static_upper_limbs(concept: Concept, out: TaskResult, t, holding: bool) -> None:
    """Quasi-static shoulder yaw and wrist loads (worst-case arm pose)."""
    g = concept.geo
    m_fh, c_fh = concept.forearm_hand_mass()
    payload = 0.5 * concept.mass.payload_hands if holding else 0.0
    yaw = m_fh * G * c_fh if holding else 0.25 * m_fh * G * c_fh
    z = np.zeros_like(t)
    out.add("shoulder_yaw", Trace("body", t, np.full_like(t, yaw), z, 1.0))
    wrist = (concept.spec["mass"]["end_effector_kg"] * 0.5 + payload) * G * g.hand
    for w in ("wrist_yaw", "wrist_pitch", "wrist_roll")[: concept.spec["dof"]["wrist"]]:
        out.add(w, Trace("body", t, np.full_like(t, wrist), z, 1.0))


def task_turn(concept: Concept, p: dict) -> TaskResult:
    g = concept.geo
    M = concept.mass.total
    rate = p["yaw_rate_rad_s"]
    # mean distance from the centre over the sole (pivot friction)
    xs = np.linspace(-g.foot_length / 2, g.foot_length / 2, 41)
    ys = np.linspace(-g.foot_width / 2, g.foot_width / 2, 21)
    X, Y = np.meshgrid(xs, ys)
    r_eq = float(np.mean(np.hypot(X, Y)))
    tau_fric = p["foot_friction"] * M * G * r_eq
    I_body = M * (0.12 * g.height) ** 2
    t_ramp = 0.3
    t = np.linspace(0, 1.0, N_SAMPLES)
    omega = rate * np.clip(t / t_ramp, 0, 1)
    alpha = np.where(t < t_ramp, rate / t_ramp, 0.0)
    out = TaskResult("turn", f"原地转向 {rate:.1f} rad/s", False, {},
                     info={"pivot_radius_m": r_eq, "yaw_inertia_kgm2": I_body})
    # the stance hip must be able to use the full foot traction, and to spin the body
    out.add("hip_yaw", Trace("body", t, np.maximum(tau_fric, I_body * alpha), omega, 1.0))
    I_up = concept.upper_body_yaw_inertia()
    out.add("waist_yaw", Trace("body", t, I_up * alpha, omega, 1.0))
    return out


def task_balance(concept: Concept, spec: dict) -> TaskResult:
    """Quasi-static ankle authority: full body weight on one foot, CoP at the edge."""
    g = concept.geo
    M = concept.mass.total
    k = spec["balance"]["ankle_cop_authority"]
    t = np.linspace(0, 1.0, 11)
    z = np.zeros_like(t)
    out = TaskResult("balance", "踝关节平衡裕度（CoP 到脚尖/脚边）", False, {},
                     info={"toe_m": g.toe, "half_width_m": g.foot_width / 2})
    out.add("ankle_pitch", Trace("toe", t, np.full_like(t, k * M * G * g.toe), z, 1.0))
    out.add("ankle_roll", Trace("toe", t, z.copy(), z.copy(), 1.0))
    out.add("ankle_roll", Trace("edge", t, np.full_like(t, k * M * G * g.foot_width / 2), z, 1.0))
    out.add("ankle_pitch", Trace("edge", t, z.copy(), z.copy(), 1.0))
    return out


def run_tasks(concept: Concept) -> list[TaskResult]:
    """Every enabled task (plus the balance authority) for one concept."""
    spec = concept.spec
    tp = spec["tasks"]
    out: list[TaskResult] = []
    if tp["stand"]["enabled"]:
        out.append(task_stand(concept, tp["stand"]))
    if tp["walk"]["enabled"]:
        out.append(task_walk(concept, tp["walk"]))
        if tp["walk"]["max_speed"] and tp["walk"]["max_speed"] > tp["walk"]["speed"]:
            out.append(task_walk(concept, tp["walk"], tp["walk"]["max_speed"], name="walk_fast"))
    if tp["stairs"]["enabled"]:
        out.append(task_stairs(concept, tp["stairs"]))
    if tp["squat_lift"]["enabled"]:
        out.append(task_squat_lift(concept, tp["squat_lift"]))
    if tp["stoop_lift"]["enabled"]:
        out.append(task_stoop_lift(concept, tp["stoop_lift"]))
    if tp["run"]["enabled"]:
        out.append(task_run(concept, tp["run"]))
    if tp["arm_lift"]["enabled"]:
        out.append(task_arm_lift(concept, tp["arm_lift"]))
    if tp["turn"]["enabled"]:
        out.append(task_turn(concept, tp["turn"]))
    out.append(task_balance(concept, spec))
    return out
