"""What the hands must reach, and the arm length that reaches it.

Postures are the ones the tasks already use, reduced to the shoulder height:

* standing — knees bent as in the ``stand`` task, trunk upright;
* low grasp — the lower of the ``stoop_lift`` posture (trunk pitched down by
  ``lean_deg``, hips slid back) and the ``squat_lift`` posture (hips at
  ``depth``, trunk leaned 25°), with the arm hanging straight down.

Constraints (each optional):

=========================  ==================================================
``floor_pick_height_m``    shoulder_low − arm ≤ value
``shelf_height_m``         shoulder_stand + arm ≥ value (arm straight up)
``forward_reach_m``        arm ≥ value
``table_height_m``         elbow_stand − 0.05 ≥ value (upper arm hanging),
                           an upper bound on the upper arm
=========================  ==================================================

"arm" is shoulder joint -> grasp point (upper arm + forearm + hand).
"""

from __future__ import annotations

import math

from .body import Geometry, stand_hip_height

TABLE_CLEARANCE = 0.05
SQUAT_LEAN_DEG = 25.0
STOOP_HIP_SHIFT = 0.3       # hip slides back by this share of the hip-ankle distance


def shoulder_heights(geo: Geometry, spec: dict) -> dict:
    """Shoulder height standing, and in the lowest grasp posture (m above the sole)."""
    t = spec["tasks"]
    stand = geo.ankle_h + (stand_hip_height(geo, t["stand"]["knee_bend_deg"]) - geo.ankle_h)
    out = {"stand": stand + geo.hip_to_shoulder}
    lows = {}
    if t["stoop_lift"]["enabled"]:
        d = stand_hip_height(geo, t["stoop_lift"]["knee_bend_deg"]) - geo.ankle_h
        hip = geo.ankle_h + d * math.sqrt(1 - STOOP_HIP_SHIFT ** 2)
        lows["stoop"] = hip + geo.hip_to_shoulder * math.cos(math.radians(t["stoop_lift"]["lean_deg"]))
    if t["squat_lift"]["enabled"] or not lows:
        hip = geo.ankle_h + t["squat_lift"]["depth"] * geo.leg
        lows["squat"] = hip + geo.hip_to_shoulder * math.cos(math.radians(SQUAT_LEAN_DEG))
    posture = min(lows, key=lows.get)
    out["low"], out["low_posture"] = lows[posture], posture
    return out


def arm_bounds(geo: Geometry, spec: dict) -> dict:
    """Shortest arm the reach constraints allow, and the upper-arm ceiling."""
    r = spec["reach"]
    sh = shoulder_heights(geo, spec)
    need = {"min_ratio": r["arm_min_ratio"] * geo.height}
    if r["floor_pick_height_m"] is not None:
        need["floor"] = sh["low"] - r["floor_pick_height_m"]
    if r["shelf_height_m"] is not None:
        need["shelf"] = r["shelf_height_m"] - sh["stand"]
    if r["forward_reach_m"] is not None:
        need["forward"] = r["forward_reach_m"]
    arm_min = max(need.values())
    binding = max(need, key=need.get)
    ua_max = None
    if r["table_height_m"] is not None:
        ua_max = sh["stand"] - r["table_height_m"] - TABLE_CLEARANCE
    return {"arm_min": arm_min, "binding": binding, "ua_max": ua_max,
            "arm_max": r["arm_max_ratio"] * geo.height, **sh}


def any_constraint(spec: dict) -> bool:
    r = spec["reach"]
    return any(r[k] is not None for k in ("floor_pick_height_m", "shelf_height_m",
                                          "table_height_m", "forward_reach_m"))


def arm_from_reach(geo: Geometry, spec: dict) -> tuple[float, float, float, str]:
    """(upper arm, forearm, hand) of the shortest arm meeting every constraint.

    Split in the proportions of the ratio defaults; if a table caps the upper
    arm, the forearm takes the rest.
    """
    b = spec["body"]
    ab = arm_bounds(geo, spec)
    arm = min(max(ab["arm_min"], 0.0), ab["arm_max"])
    w = (b["upper_arm_ratio"], b["forearm_ratio"], b["hand_ratio"])
    tot = sum(w)
    ua, fa, hd = (arm * x / tot for x in w)
    note = f"手臂长度由够取约束决定（{_CN.get(ab['binding'], ab['binding'])}）"
    if ab["ua_max"] is not None and ua > ab["ua_max"] > 0:
        fa += ua - ab["ua_max"]
        ua = ab["ua_max"]
        note += "；上臂受工作台高度限制，余量给前臂"
    return ua, fa, hd, note


_CN = {"floor": "地面拾取", "shelf": "货架高度", "forward": "前伸距离",
       "min_ratio": "最小臂长比例"}


def check_reach(geo: Geometry, spec: dict) -> list[dict]:
    """One check per stated constraint, against the concept's actual arm."""
    r = spec["reach"]
    sh = shoulder_heights(geo, spec)
    arm = geo.upper_arm + geo.forearm + geo.hand
    out = []

    def add(name, value, limit, ok, note):
        out.append({"name": name, "value": round(value, 3), "limit": limit, "unit": "m",
                    "ok": ok, "severity": "error", "note": note})
    if r["floor_pick_height_m"] is not None:
        low = sh["low"] - arm
        add("够取：地面拾取", low, r["floor_pick_height_m"], low <= r["floor_pick_height_m"] + 1e-6,
            f"{'弯腰' if sh['low_posture'] == 'stoop' else '深蹲'}姿态手最低 {low:.2f} m，要求 ≤ {r['floor_pick_height_m']:.2f} m")
    if r["shelf_height_m"] is not None:
        top = sh["stand"] + arm
        add("够取：货架高度", top, r["shelf_height_m"], top >= r["shelf_height_m"] - 1e-6,
            f"站立举手最高 {top:.2f} m，要求 ≥ {r['shelf_height_m']:.2f} m")
    if r["forward_reach_m"] is not None:
        add("够取：前伸距离", arm, r["forward_reach_m"], arm >= r["forward_reach_m"] - 1e-6,
            f"臂长（肩—抓取点）{arm:.2f} m，要求 ≥ {r['forward_reach_m']:.2f} m")
    if r["table_height_m"] is not None:
        elbow = sh["stand"] - geo.upper_arm
        ok = elbow - TABLE_CLEARANCE >= r["table_height_m"] - 1e-6
        add("够取：工作台高度", elbow, r["table_height_m"], ok,
            f"站立肘高 {elbow:.2f} m，台面 {r['table_height_m']:.2f} m 需低于肘 ≥ {TABLE_CLEARANCE*100:.0f} cm")
    arm_max = r["arm_max_ratio"] * geo.height
    if arm > arm_max + 1e-6:
        add("够取：臂长上限", arm, round(arm_max, 3), False,
            f"需要臂长 {arm:.2f} m，超过 {r['arm_max_ratio']:.2f}×身高 = {arm_max:.2f} m")
    return out
