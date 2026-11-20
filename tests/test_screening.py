"""The concept screening: dynamics against closed forms, selection invariants,
and an end-to-end concept that must close near the published population."""

import math

import numpy as np
import pytest

from draft.screening.actuators import (
    SlotRequirement, TaskDemand, _motor_traces, available_torque, envelope_ratio,
    match_catalog, size_by_trend)
from draft.screening.body import actuator_slots, build_concept, geometry
from draft.screening.planar import G, PlanarBody, PlanarTree, angle_series
from draft.screening.spec import make_spec
from draft.screening.tasks import Trace, leg_ik, pendulum_x


# ── planar inverse dynamics ────────────────────────────────────────────────────

def test_static_pendulum_holds_its_weight():
    m, L = 3.0, 0.4
    tree = PlanarTree([PlanarBody("link", -1, (0, 0), "j", m, (L, 0.0), 0.0)])
    for phi in (0.0, 0.7, -1.2):
        a = {"link": np.array([phi])}
        z = {"link": np.zeros(1)}
        tau = tree.inverse_dynamics(a, z, z).tau["j"][0]
        assert tau == pytest.approx(m * G * L * math.cos(phi), rel=1e-12)


def test_double_pendulum_matches_lagrange():
    """RNEA against the closed-form equations of motion of a double pendulum."""
    m1, m2, l1, c1, c2, I1, I2 = 2.0, 1.3, 0.5, 0.2, 0.25, 0.03, 0.02
    tree = PlanarTree([
        PlanarBody("a", -1, (0, 0), "q1", m1, (c1, 0), I1),
        PlanarBody("b", 0, (l1, 0), "q2", m2, (c2, 0), I2),
    ])
    rng = np.random.default_rng(0)
    for _ in range(5):
        q1, q2, d1, d2, a1, a2 = rng.uniform(-2, 2, 6)
        phi = {"a": np.array([q1]), "b": np.array([q1 + q2])}
        dphi = {"a": np.array([d1]), "b": np.array([d1 + d2])}
        ddphi = {"a": np.array([a1]), "b": np.array([a1 + a2])}
        tau = tree.inverse_dynamics(phi, dphi, ddphi).tau
        # M(q) qdd + C(q, qd) qd + g(q), absolute-angle form converted to relative
        c = math.cos(q2)
        M11 = I1 + I2 + m1 * c1 ** 2 + m2 * (l1 ** 2 + c2 ** 2 + 2 * l1 * c2 * c)
        M12 = I2 + m2 * (c2 ** 2 + l1 * c2 * c)
        M22 = I2 + m2 * c2 ** 2
        h = m2 * l1 * c2 * math.sin(q2)
        C1 = -h * (2 * d1 * d2 + d2 ** 2)
        C2 = h * d1 ** 2
        g1 = (m1 * c1 + m2 * l1) * G * math.cos(q1) + m2 * c2 * G * math.cos(q1 + q2)
        g2 = m2 * c2 * G * math.cos(q1 + q2)
        assert tau["q1"][0] == pytest.approx(M11 * a1 + M12 * a2 + C1 + g1, rel=1e-9)
        assert tau["q2"][0] == pytest.approx(M12 * a1 + M22 * a2 + C2 + g2, rel=1e-9)


def test_armature_adds_reflected_inertia():
    tree = PlanarTree([PlanarBody("a", -1, (0, 0), "j", 0.0, (0, 0), 0.0, armature=0.5)])
    z = {"a": np.zeros(1)}
    tau = tree.inverse_dynamics({"a": np.array([0.3])}, z, {"a": np.array([4.0])}).tau["j"][0]
    assert tau == pytest.approx(2.0)


def test_angle_series_differentiates_smooth_signals():
    t = np.linspace(0, 1, 401)
    phi, dphi, ddphi = angle_series(t, {"x": np.sin(3 * t)})
    assert np.allclose(dphi["x"][5:-5], 3 * np.cos(3 * t[5:-5]), atol=1e-3)
    assert np.allclose(ddphi["x"][5:-5], -9 * np.sin(3 * t[5:-5]), atol=1e-2)


# ── kinematics and reference motions ───────────────────────────────────────────

def test_leg_ik_closes_and_bends_the_knee_forward():
    hip = np.array([[0.05, 0.6], [-0.1, 0.5]])
    ankle = np.zeros((2, 2))
    knee, reach = leg_ik(hip, ankle, 0.35, 0.33)
    assert np.allclose(np.linalg.norm(knee - hip, axis=1), 0.35)
    assert np.allclose(np.linalg.norm(knee - ankle, axis=1), 0.33)
    # knee ahead of the hip-ankle line
    assert np.all(knee[:, 0] > (hip[:, 0] + ankle[:, 0]) / 2)
    assert reach < 1


def test_pendulum_with_constant_height_is_the_lipm():
    t = np.linspace(0, 0.5, 301)
    z = 0.8
    x = pendulum_x(t, np.full_like(t, z), -0.2, 0.2, 0.0)
    Tc = math.sqrt(z / G)
    ref = 0.2 * np.sinh((t - 0.25) / Tc) / math.sinh(0.25 / Tc)
    assert np.allclose(x, ref, atol=1e-6)


# ── actuators ───────────────────────────────────────────────────────────────────

def test_parallel_ankle_split_conserves_power():
    t = np.linspace(0, 1, 50)
    tp, tr = 30 * np.sin(t), 10 * np.cos(t)
    wp, wr = 2 * np.cos(t), -np.sin(3 * t)
    traces = {"ankle_pitch": [Trace("stance", t, tp, wp, 0.5)],
              "ankle_roll": [Trace("stance", t, tr, wr, 0.5)]}
    (A,), (B,) = _motor_traces(traces, "ankle_drive", True)
    assert np.allclose(A[0] * A[1] + B[0] * B[1], tp * wp + tr * wr)


def test_envelope_k1_is_the_catalogue_straight_line():
    w = np.array([0.0, 5.0, 10.0])
    assert np.allclose(available_torque(100, 10, w, 1.0), [100, 50, 0])
    assert np.allclose(available_torque(100, 10, w, 2.0), [100, 100, 0])
    assert envelope_ratio(100, 10, np.array([50.0]), np.array([5.0]), 1.0) == pytest.approx(1.0)


def _req(tau, omega, rms=0.0):
    rq = SlotRequirement("knee")
    tau, omega = np.asarray(tau, float), np.asarray(omega, float)
    rq.demands.append(TaskDemand("x", "x", True, tau, omega, tau * omega, rms))
    rq.thermal_rms = rms
    return rq


def test_trend_actuator_covers_its_requirement():
    spec = make_spec()
    rq = _req([120, 60, 10], [0.5, 8.0, 14.0], rms=30)
    a = size_by_trend(rq, spec)
    assert a.feasible and a.torque_margin >= -1e-9 and a.thermal_margin >= -1e-9
    # bigger demand, heavier actuator
    assert size_by_trend(_req([240, 60, 10], [0.5, 8.0, 14.0], 30), spec).mass > a.mass


def test_catalogue_matches_are_feasible_and_lightest_first():
    spec = make_spec()
    rq = _req([20, 10], [1.0, 10.0], rms=5)
    parts = match_catalog(rq, spec)
    assert parts, "a 20 Nm joint must find catalogue parts"
    assert all(p.feasible for p in parts)
    assert [p.mass for p in parts] == sorted(p.mass for p in parts)


# ── spec, geometry, the whole loop ─────────────────────────────────────────────

def test_spec_rejects_unknown_keys_and_bad_proportions():
    with pytest.raises(KeyError):
        make_spec({"body": {"heigth_m": 1.5}})
    with pytest.raises(ValueError):
        make_spec({"body": {"leg_ratio": 0.62, "head_neck_ratio": 0.3}})


def test_geometry_adds_up_to_the_height():
    g = geometry(make_spec({"body": {"height_m": 1.6}}))
    assert g.ankle_h + g.leg + g.hip_to_shoulder + g.head_neck == pytest.approx(1.6)


def test_parallel_ankle_has_four_drive_motors():
    spec = make_spec()
    slots = actuator_slots(spec)
    c = build_concept(spec, {s: 1.0 for s in slots}, {s: 0.0 for s in slots}, 2.0)
    assert "ankle_drive" in slots and "ankle_pitch" not in slots
    assert c.mass.actuator_count["ankle_drive"] == 4


@pytest.fixture(scope="module")
def concept_1p5():
    from draft.screening.sizing import evaluate
    return evaluate(make_spec({"body": {"height_m": 1.5}}))


def test_default_concept_closes_near_the_population(concept_1p5):
    from draft.trends import link_fits
    r = concept_1p5
    assert r.closed and r.converged
    mv = link_fits("humanoid").mass_vs_height
    trend = mv["coef"] * r.concept.geo.z_span ** mv["exp"]
    assert 0.7 < r.mass / trend < 1.4
    # every slot got an actuator that covers it
    assert all(a.torque_margin >= -1e-6 for a in r.chosen.values())
    # the knee is the most loaded leg joint, as in every published humanoid
    assert r.req["knee"].peak_tau == max(r.req[s].peak_tau for s in r.req)


def test_export_names_a_motor_class_for_every_slot(concept_1p5):
    from draft.screening.export import CLASS_OF, draft_overrides
    p = draft_overrides(concept_1p5)
    for slot in concept_1p5.chosen:
        cls = CLASS_OF[slot][0]
        assert p[f"{cls}_motor_effort"] > 0
        for role in CLASS_OF[slot][1]:
            assert p[f"{role}_mot"] == cls
    assert p["torso_roll_dof"] is False and "torso_yaw_dof" not in p


def test_sweep_grid_is_the_full_product():
    from draft.screening.sweep import grid
    g = grid(make_spec(), {"body.height_m": [1.2, 1.5], "body.leg_ratio": [0.4, 0.45, 0.5]})
    assert len(g) == 6
    assert all(not isinstance(s, Exception) for _, s in g)
