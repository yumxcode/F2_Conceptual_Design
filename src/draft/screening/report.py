"""HTML reports (Chinese text, English chart labels) for a concept, a sweep and
the benchmark. Charts are matplotlib PNGs embedded as data URIs, so a report is
one self-contained file that opens anywhere."""

from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path

import numpy as np

from .actuators import available_torque
from .reference import load_reference
from .spec import dof_count

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#9a9993"
INK = "#0b0b0b"
INK2 = "#52514e"

_CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e2dc;
--ok:#0d7a3a;--warn:#a45d00;--bad:#b42318;--accent:#2a78d6;--chip:#f3f2ee}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#222220;
--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--ok:#4cc27a;--warn:#e0a040;--bad:#f07167;
--accent:#3987e5;--chip:#2c2b29}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#222220;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;
--ok:#4cc27a;--warn:#e0a040;--bad:#f07167;--accent:#3987e5;--chip:#2c2b29}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:8px 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-top:8px;
border-top:1px solid var(--line)}h3{font-size:16px;margin:20px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{color:var(--ink2);font-size:13px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--chip);color:var(--ink2);font-weight:600;position:sticky;top:0}
td:first-child,th:first-child{text-align:left}td.l{text-align:left}
tr:last-child td{border-bottom:none}
.ok{color:var(--ok);font-weight:600}.warn{color:var(--warn);font-weight:600}.bad{color:var(--bad);font-weight:600}
img{max-width:100%;height:auto;background:#fff;border-radius:10px;border:1px solid var(--line)}
.note{background:var(--chip);border-radius:10px;padding:10px 14px;color:var(--ink2);font-size:14px}
ul{padding-left:20px}code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": "#c9c7c0", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": "#ecebe6", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 110,
    })
    return plt


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    _plt().close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _e(x) -> str:
    return html.escape(str(x))


def _f(x, nd=1) -> str:
    if x is None:
        return "–"
    if isinstance(x, float) and not math.isfinite(x):
        return "∞"
    return f"{x:.{nd}f}"


def _table(head: list, rows: list, left_cols: int = 1) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)   # headers are trusted literals
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = ' class="l"' if i < left_cols else ""
            tds.append(f"<td{cls}>{c}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _page(title: str, body: str) -> str:
    return (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>{body}</main></body></html>")


SLOT_CN = {"hip_pitch": "髋俯仰", "hip_roll": "髋横滚", "hip_yaw": "髋偏航", "knee": "膝",
           "ankle_pitch": "踝俯仰", "ankle_roll": "踝横滚", "ankle_drive": "踝（并联驱动电机）",
           "waist_yaw": "腰偏航", "waist_roll": "腰横滚", "waist_pitch": "腰俯仰",
           "shoulder_pitch": "肩俯仰", "shoulder_roll": "肩横滚", "shoulder_yaw": "肩偏航",
           "elbow": "肘", "wrist_yaw": "腕偏航", "wrist_pitch": "腕俯仰", "wrist_roll": "腕横滚"}
TASK_EN = {"walk": "walk", "walk_fast": "walk fast", "stairs": "stairs", "squat_lift": "squat lift",
           "stoop_lift": "stoop lift",
           "run": "run", "stand": "stand", "arm_lift": "arm lift", "turn": "turn",
           "balance": "ankle authority", "agility": "agility floor"}


# ── charts ─────────────────────────────────────────────────────────────────────

def chart_torque_speed(res, slots: list) -> str:
    plt = _plt()
    n = len(slots)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 3.0 * rows), squeeze=False)
    main_tasks = ["walk", "walk_fast", "stairs", "squat_lift", "stoop_lift", "run"]
    color = {t: SERIES[i] for i, t in enumerate(main_tasks)}
    sf = res.spec["actuators"]["torque_safety_factor"]
    k = res.spec["actuators"]["voltage_limit_factor"]
    for ax, slot in zip(axes.flat, slots):
        rq = res.req[slot]
        for d in rq.demands:
            c = color.get(d.task, MUTED)
            lab = TASK_EN.get(d.task, d.task) if d.task in color else "other"
            ax.scatter(d.omega, d.tau * sf, s=4, color=c, alpha=0.55, linewidths=0,
                       label=lab, rasterized=True)
        for act, style, col in ((res.chosen[slot], "-", INK), (res.trend[slot], "--", MUTED)):
            if not math.isfinite(act.mass) or act.omega_nl <= 0:
                continue
            w = np.linspace(0, act.omega_nl, 100)
            ax.plot(w, available_torque(act.tau_peak, act.omega_nl, w, k), style, color=col, lw=1.6)
        ax.set_title(f"{slot}  ({res.chosen[slot].name})", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("speed |ω| (rad/s)")
        ax.set_ylabel(f"|τ| × SF {sf:g} (Nm)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    seen = {}
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    order = [TASK_EN[t] for t in main_tasks] + ["other"]
    l = [x for x in order if x in seen]
    h = [seen[x] for x in l]
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=INK, lw=1.6), Line2D([], [], color=MUTED, lw=1.6, ls="--")]
    l += ["chosen actuator envelope", "trend actuator envelope"]
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 4), markerscale=3,
               bbox_to_anchor=(0.5, 1.0 + 0.02 / rows))
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.02 * (rows == 1)))
    return _png(fig)


def chart_mass(res) -> str:
    plt = _plt()
    b = res.concept.mass.breakdown()
    names = {"actuators": "actuators", "limb_structure": "limb structure",
             "trunk_shell": "trunk shell", "battery": "battery", "head": "head shell",
             "compute": "compute", "sensors": "sensors", "end_effectors": "hands"}
    items = sorted(b.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.barh([names[k] for k, _ in items], [v for _, v in items], color=SERIES[0], height=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v:.1f} kg", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("mass (kg)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v for _, v in items) * 1.25)
    return _png(fig)


def chart_population(points: list[dict], refs=None) -> str:
    """Mass vs height: survey robots, Draft's trend, and the given concepts."""
    plt = _plt()
    from ..trends import link_fits
    from .reference import Z_SPAN_SHARE
    refs = refs if refs is not None else load_reference()
    mv = link_fits("humanoid").mass_vs_height
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    H = np.array([r.height_est_m for r in refs])
    M = np.array([r.mass_kg for r in refs])
    ax.scatter(H, M, s=22, color=MUTED, label="published humanoids (survey)", zorder=2)
    hh = np.linspace(0.8, 2.1, 100)
    ax.plot(hh, mv["coef"] * (hh * Z_SPAN_SHARE) ** mv["exp"], color=MUTED, lw=1.2, ls="--",
            label="Draft mass-vs-size trend")
    for i, p in enumerate(points):
        ax.scatter([p["height_m"]], [p["mass_kg"]], s=60, color=SERIES[i % 8], zorder=3,
                   edgecolors="white", linewidths=1.5, label=p.get("label", "concept"))
    ax.set_xlabel("standing height (m; survey robots estimated from joint span)")
    ax.set_ylabel("total mass (kg)")
    ax.legend(fontsize=8)
    return _png(fig)


def chart_heatmap(rows: list, xkey: str, ykey: str, value: str, label: str,
                  fmt: str = "{:.0f}") -> str:
    plt = _plt()
    xs = sorted({r[xkey] for r in rows})
    ys = sorted({r[ykey] for r in rows})
    Z = np.full((len(ys), len(xs)), np.nan)
    bad = np.zeros_like(Z, dtype=bool)
    for r in rows:
        i, j = ys.index(r[ykey]), xs.index(r[xkey])
        v = r.get(value)
        if isinstance(v, (int, float)) and math.isfinite(v):
            Z[i, j] = v
        bad[i, j] = not r.get("feasible", False)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(xs), 0.9 + 0.55 * len(ys)))
    im = ax.imshow(Z, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    ax.set_xlabel(xkey.split(".")[-1])
    ax.set_ylabel(ykey.split(".")[-1])
    ax.grid(False)
    vmin = np.nanmin(Z) if np.isfinite(Z).any() else 0
    vmax = np.nanmax(Z) if np.isfinite(Z).any() else 1
    span = max(vmax - vmin, 1e-9)
    for i in range(len(ys)):
        for j in range(len(xs)):
            if np.isfinite(Z[i, j]):
                dark = (Z[i, j] - vmin) / span > 0.55
                txt = fmt.format(Z[i, j]) + (" ✕" if bad[i, j] else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if dark else INK)
            else:
                ax.text(j, i, "✕", ha="center", va="center", fontsize=9, color="#b42318")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(label)
    ax.set_title(label + "   (✕ = not closed / an actuator beyond the frontier)", fontsize=8,
                 loc="left", color=INK2)
    return _png(fig)


def chart_benchmark(rows: list) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ok = [r for r in rows if r["closed"]]
    ax = axes[0]
    real = [r["mass_real_kg"] for r in ok]
    pred = [r["mass_pred_kg"] for r in ok]
    ax.scatter(real, pred, s=26, color=SERIES[0], zorder=3)
    lim = [0, max(real + pred) * 1.1]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    for r in ok:
        ax.annotate(r["robot"], (r["mass_real_kg"], r["mass_pred_kg"]), fontsize=6.5,
                    color=INK2, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("published mass (kg)")
    ax.set_ylabel("screening mass (kg)")
    ax.set_title("total mass", loc="left", fontsize=9)
    ax = axes[1]
    for i, cls in enumerate(("knee", "hip_pitch", "ankle_pitch")):
        pts = [(r[f"{cls}_real_Nm"], r[f"{cls}_pred_Nm"]) for r in ok
               if r.get(f"{cls}_real_Nm") and r.get(f"{cls}_pred_Nm")]
        if pts:
            a = np.array(pts)
            ax.scatter(a[:, 0], a[:, 1], s=22, color=SERIES[i], label=cls, zorder=3)
    m = max([r.get("knee_real_Nm") or 0 for r in ok] + [r.get("knee_pred_Nm") or 0 for r in ok]) * 1.1
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("published joint limit (Nm)")
    ax.set_ylabel("screening peak demand, no safety factor (Nm)")
    ax.set_title("joint torque: demand vs what vendors installed", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _png(fig)


# ── concept report ─────────────────────────────────────────────────────────────

def _status(res) -> tuple[str, str]:
    if not res.closed:
        return "bad", "无法闭合"
    errs = [w for w in res.warnings() if w.get("severity") == "error"]
    if errs:
        return "bad", "有执行器超出产品能力"
    if res.warnings():
        return "warn", "可行（有提示）"
    return "ok", "可行"


def concept_report(res, export_path: str | None = None) -> str:
    spec = res.spec
    g = res.concept.geo
    e = res.energy
    cls, label = _status(res)
    b = []
    b.append(f"<h1>人形机器人概念粗筛：{_e(spec['name'])}</h1>")
    b.append(f'<p class="sub">身高 {g.height:.2f} m · 腿长比 {spec["body"]["leg_ratio"]:.2f} · '
             f'{dof_count(spec)} 自由度 · 选型模式：{"目录真实模组" if spec["actuators"]["mode"] == "catalog" else "趋势虚拟模组"} · '
             f'状态：<span class="{cls}">{label}</span></p>')
    tiles = [
        (f"{res.mass:.1f} kg" if res.closed else "∞", "整机质量"),
        (f"{res.concept.mass.breakdown()['actuators']:.1f} kg", "执行器总质量"),
        (f"{e.battery_Wh:.0f} Wh", f"电池（{e.battery_kg:.1f} kg）"),
        (f"{spec['electrical']['runtime_h']:g} h", "目标续航（任务剖面）"),
        (f"{e.mission_W:.0f} W", "平均电功率"),
        (f"{e.peak_W/1000:.2f} kW", f"峰值功率（{_e(e.peak_task)}）"),
        (f"{e.cot_walk:.2f}", "步行运输成本 CoT"),
    ]
    b.append('<div class="tiles">' + "".join(f"<div class='tile'><b>{v}</b><span>{k}</span></div>"
                                             for v, k in tiles) + "</div>")
    if res.notes:
        b.append('<div class="note">' + "<br>".join(_e(n) for n in res.notes) + "</div>")

    # warnings
    ws = res.warnings()
    b.append("<h2>检查与提示</h2>")
    if not ws:
        b.append('<p class="ok">所有检查通过。</p>')
    else:
        rows = []
        for w in ws:
            sev = w.get("severity", "error")
            c = "bad" if sev == "error" else "warn"
            rows.append([f'<span class="{c}">{"错误" if c == "bad" else "提示"}</span>',
                         _e(w["name"]), f'<span style="white-space:normal">{_e(w.get("note", ""))}</span>'])
        b.append(_table(["级别", "检查", "说明"], rows, left_cols=3))
        b.append('<p class="sub">CoP（压力中心）超出脚底是“提示”：参考动作没有利用躯干和手臂摆动来平衡，'
                 '说明这一步态下脚偏小或步态偏激进，可加大 <code>body.foot_length_ratio</code> 或降低速度。</p>')

    # actuators
    b.append("<h2>关节需求与执行器选型</h2>")
    b.append(f'<p class="sub">需求已乘安全系数：扭矩 ×{spec["actuators"]["torque_safety_factor"]:g}、'
             f'转速 ×{spec["actuators"]["speed_safety_factor"]:g}、发热 ×{spec["actuators"]["thermal_safety_factor"]:g}。'
             f'裕度 = 可用/需求 − 1。转矩–转速包络：电流限 + 电压线（k = {spec["actuators"]["voltage_limit_factor"]:g}）。</p>')
    rows = []
    for s, rq in res.req.items():
        a = res.chosen[s]
        n = res.concept.mass.actuator_count[s]
        fe = '<span class="ok">✓</span>' if a.feasible else '<span class="bad">✕</span>'
        rows.append([
            f"{SLOT_CN.get(s, s)}<br><small>{s} ×{n}</small>",
            f"{rq.peak_tau:.1f}<br><small>{_e(rq.governing('tau'))}</small>",
            f"{rq.peak_omega:.1f}<br><small>{_e(rq.governing('omega'))}</small>",
            f"{rq.peak_power:.0f}",
            f"{rq.thermal_rms:.1f}<br><small>{_e(rq.thermal_source)}</small>",
            f'<span style="white-space:normal">{_e(a.name)}</span><br><small>{_e(a.vendor or a.source)}</small>',
            f"{a.tau_peak:.0f} / {a.omega_nl:.1f}",
            f"{a.mass:.2f}",
            f"{a.torque_margin:+.0%}", f"{a.thermal_margin:+.0%}" if a.thermal_margin < 98 else "∞", fe,
        ])
    b.append(_table(["关节", "峰值扭矩 Nm<br>(工况)", "峰值转速 rad/s", "峰值功率 W", "发热 RMS Nm",
                     "选定模组", "峰值 Nm / 空载 rad/s", "质量 kg", "扭矩裕度", "热裕度", "可行"], rows))
    b.append("<h3>各关节目录候选（按质量升序）</h3>")
    rows = []
    for s, cands in res.candidates.items():
        t = res.trend[s]
        trend_txt = f"{t.tau_peak:.0f} Nm, N={t.gear:.0f}, {t.mass:.2f} kg" + ("" if t.feasible else " ✕")
        cand_txt = "<br>".join(f"{_e(c.name)}（{_e(c.vendor)}，{c.tau_peak:.0f} Nm，{c.mass:.2f} kg，裕度 {c.torque_margin:+.0%}）"
                               for c in cands) or '<span class="warn">目录中无满足的模组</span>'
        rows.append([SLOT_CN.get(s, s), trend_txt, f'<span style="white-space:normal">{cand_txt}</span>'])
    b.append(_table(["关节", "趋势最优（假想模组）", "目录候选"], rows, left_cols=3))
    leg = [s for s in res.req if s in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_drive",
                                        "ankle_pitch", "ankle_roll", "shoulder_pitch", "elbow")]
    b.append("<h3>扭矩–转速工作点与包络</h3>")
    b.append(f'<img alt="torque-speed" src="{chart_torque_speed(res, leg)}">')

    # geometry & mass
    b.append("<h2>几何与质量</h2>")
    geo_rows = [["身高", f"{g.height:.3f}"], ["髋—踝（腿长）", f"{g.leg:.3f}"],
                ["大腿 / 小腿", f"{g.thigh:.3f} / {g.shank:.3f}"], ["踝高", f"{g.ankle_h:.3f}"],
                ["髋—肩", f"{g.hip_to_shoulder:.3f}"], ["肩—头顶", f"{g.head_neck:.3f}"],
                ["上臂 / 前臂 / 手", f"{g.upper_arm:.3f} / {g.forearm:.3f} / {g.hand:.3f}"],
                ["脚长 / 脚宽（踝前 {:.3f}）".format(g.toe), f"{g.foot_length:.3f} / {g.foot_width:.3f}"],
                ["髋宽 / 肩宽（关节间）", f"{g.hip_width:.3f} / {g.shoulder_width:.3f}"]]
    b.append('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px">'
             + "<div>" + _table(["尺寸", "m"], geo_rows) + "</div>"
             + f'<div><img alt="mass" src="{chart_mass(res)}"></div></div>')
    b.append("<h3>放在已发布人形机器人中看</h3>")
    b.append(f'<img alt="population" src="{chart_population([{"height_m": g.height, "mass_kg": res.mass, "label": spec["name"]}])}">')

    # energy
    b.append("<h2>能耗与电池</h2>")
    rows = []
    for name, tp in e.tasks.items():
        if name == "balance":
            continue
        share = next((m["share"] for m in spec["mission"] if m["task"] == name), 0)
        rows.append([_e(tp.label), f"{tp.mech_positive_W:.0f}", f"{tp.copper_W:.0f}",
                     f"{tp.electrical_W:.0f}", f"{tp.peak_W:.0f}", f"{share:.0%}" if share else "–"])
    b.append(_table(["工况", "正机械功率 W", "铜损 W", "平均电功率 W", "峰值电功率 W", "任务剖面占比"], rows))
    el = spec["electrical"]
    b.append(f'<p>电池：{e.battery_Wh:.0f} Wh（含 {el["battery_reserve"]:.0%} 储备），'
             f'{e.battery_kg:.2f} kg @ {el["battery_Wh_per_kg"]:g} Wh/kg，约 {e.battery_L:.2f} L。'
             f'峰值功率 {e.peak_W:.0f} W → {el["bus_voltage_V"]:g} V 母线峰值电流 {e.peak_current_A:.0f} A，'
             f'需要 {e.c_rate_needed:.1f}C 放电（设定上限 {el["battery_max_C"]:g}C，'
             f'<span class="{"ok" if e.c_rate_ok else "warn"}">{"满足" if e.c_rate_ok else "不足"}</span>）。'
             f'步行 {spec["tasks"]["walk"]["speed"]:g} m/s 的运输成本 CoT = {e.cot_walk:.2f}'
             f'（仅执行器 {e.cot_walk_actuators:.2f}）。</p>')

    # tasks
    b.append("<h2>工况参数</h2>")
    rows = []
    for r in res.tasks:
        info = "，".join(f"{k}={_f(v, 3) if isinstance(v, float) else v}" for k, v in r.info.items())
        rows.append([_e(r.label), "持续" if r.continuous else "间歇", f'<span style="white-space:normal">{_e(info)}</span>'])
    b.append(_table(["工况", "热负荷", "参数"], rows, left_cols=3))

    if export_path:
        b.append("<h2>交给 Draft 生成 MJCF</h2>")
        b.append(f"<p>已写出 <code>{_e(export_path)}</code>。在装有 MuJoCo 的机器上运行：</p>"
                 f"<p><code>python scripts/generate_robot.py --robot humanoid --overrides {_e(export_path)}</code></p>")
    b.append(_assumptions())
    return _page(f"概念粗筛 {spec['name']}", "".join(b))


def _assumptions() -> str:
    items = [
        "平面模型：矢状面（俯仰关节）与冠状面（横滚关节）分别做逆动力学；偏航、腕部等关节用准静态估算。",
        "参考动作而非控制器：步行/上下楼/跑步用倒立摆质心轨迹 + 腿部逆运动学；未利用躯干/手臂摆动平衡，CoP 提示偏保守。",
        "质量：执行器质量 m=0.0653·τ^0.666（Draft 拟合 114 款），四肢结构质量取 Draft 人形结构趋势，躯干壳体按整机质量比例；电池、计算、传感器单列。",
        "转矩–转速：电流限 + 电压线；持续扭矩 = 峰值 / 类别系数（目录未给额定值时）。",
        "铜损用热容量假设：在持续扭矩下每公斤执行器发热 copper_loss_W_per_kg。",
        "峰值功率按同一时刻同时工作的关节逐点求和（双腿支撑/摆动同时计入）。",
        "并联踝：两电机等分俯仰与横滚，连杆比按 1 处理。",
        "这些是概念阶段的数量级判断，不能代替详细设计、多体仿真和样机测试。",
    ]
    return "<h2>假设与局限</h2><ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


# ── sweep & benchmark reports ──────────────────────────────────────────────────

def sweep_report(base: dict, axes: dict, rows: list, ranked: list,
                 details: list | None = None) -> str:
    keys = list(axes)
    b = [f"<h1>参数扫描：{_e(base['name'])}</h1>",
         f'<p class="sub">{len(rows)} 个方案 · 扫描维度：' + "，".join(f"<code>{_e(k)}</code> = {_e(v)}" for k, v in axes.items())
         + f' · 可行 {len(ranked)} 个</p>']
    if details:
        b.append(f"<h2>按 {_e(keys[0])} 的推荐方案（目录真实模组）</h2>")
        b.append('<p class="sub">每个取值下最轻的可行组合，再用 114 款目录模组（及自定义清单）闭合质量环。'
                 '点击报告查看完整选型。</p>')
        slots = list(details[0]["parts"])
        head = keys + ["质量 kg", "电池 Wh", "CoT"] + [SLOT_CN.get(x, x) for x in slots] + ["报告"]
        t = []
        for d in details:
            c = d["catalog"]
            cells = [f"{d['point'][k]}" for k in keys] + [
                _f(c["mass_kg"]), f"{c['battery_Wh']:.0f}", f"{c['cot_walk']:.2f}"]
            for x in slots:
                name, vendor, m, ok = d["parts"][x]
                cls = "" if ok else ' class="bad"'
                cells.append(f"<span{cls}>{_e(name)}</span><br><small>{m:.2f} kg</small>")
            cells.append(f'<a href="{_e(d["report"])}">打开</a>')
            t.append(cells)
        b.append(_table(head, t, left_cols=len(keys)))
        b.append('<p class="sub">“趋势模组”表示目录中没有能覆盖需求的产品，按 Draft 趋势给出假想模组规格（峰值扭矩/减速比），'
                 '可作为向供应商询价或定制的规格。</p>')
    if ranked:
        b.append(_tradeoff_text(ranked, keys))
    if len(keys) >= 2:
        x, y = keys[0], keys[1]
        sub = rows
        if len(keys) > 2:
            fix = {k: rows[0][k] for k in keys[2:]}
            sub = [r for r in rows if all(r.get(k) == v for k, v in fix.items())]
            b.append(f'<p class="sub">热力图固定：{_e(fix)}</p>')
        for value, label, fmt in (("mass_kg", "total mass (kg)", "{:.0f}"),
                                  ("knee_tau_Nm", "knee peak demand (Nm)", "{:.0f}"),
                                  ("hip_pitch_tau_Nm", "hip pitch peak demand (Nm)", "{:.0f}"),
                                  ("battery_Wh", "battery (Wh)", "{:.0f}"),
                                  ("cot_walk", "walking cost of transport", "{:.2f}")):
            if any(value in r for r in sub):
                b.append(f'<h3>{_e(label)}</h3><img alt="{_e(value)}" src="{chart_heatmap(sub, x, y, value, label, fmt)}">')
    pts = [{"height_m": r["height_m"], "mass_kg": r["mass_kg"], "label": "screened concepts"}
           for r in rows if math.isfinite(r.get("mass_kg", math.inf))]
    if pts:
        b.append("<h3>放在已发布人形机器人中看</h3>")
        b.append(f'<img alt="population" src="{_population_many(pts)}">')
    b.append("<h2>全部可行方案（按质量；★ = 质量与步行能耗的 Pareto 前沿）</h2>")
    head = keys + ["质量 kg", "执行器 kg", "电池 Wh", "CoT", "膝 Nm", "髋俯仰 Nm", "踝 Nm", "Pareto", "提示"]
    ank = "ankle_drive_tau_Nm" if any("ankle_drive_tau_Nm" in r for r in ranked) else "ankle_pitch_tau_Nm"
    rows_t = []
    for r in ranked[:60]:
        rows_t.append([f"{r[k]}" for k in keys] + [
            f"{r['mass_kg']:.1f}", f"{r['actuator_kg']:.1f}", f"{r['battery_Wh']:.0f}",
            f"{r['cot_walk']:.2f}", f"{r.get('knee_tau_Nm', 0):.0f}", f"{r.get('hip_pitch_tau_Nm', 0):.0f}",
            f"{r.get(ank, 0):.0f}", "★" if r.get("pareto") else "",
            f'<span style="white-space:normal">{_e(r.get("warnings", ""))}</span>'])
    b.append(_table(head, rows_t, left_cols=len(keys)) if rows_t else '<p class="warn">没有可行方案。</p>')
    bad = [r for r in rows if not r.get("feasible")]
    if bad:
        b.append("<h3>不可行方案</h3>")
        b.append(_table(keys + ["原因"], [[f"{r[k]}" for k in keys] + [
            f'<span style="white-space:normal">{_e(r.get("error") or r.get("warnings") or "未闭合")}</span>']
            for r in bad], left_cols=len(keys) + 1))
    b.append(_assumptions())
    return _page(f"参数扫描 {base['name']}", "".join(b))


def _tradeoff_text(ranked: list, keys: list) -> str:
    """A few sentences on how each swept parameter moves mass, knee demand and CoT."""
    import statistics as st
    out = []
    for k in keys:
        vals = sorted({r[k] for r in ranked})
        if len(vals) < 2:
            continue
        lo = [r for r in ranked if r[k] == vals[0]]
        hi = [r for r in ranked if r[k] == vals[-1]]
        def m(rs, f):
            return st.mean(r[f] for r in rs)
        out.append(
            f"<li><code>{_e(k)}</code> 从 {vals[0]} 到 {vals[-1]}：平均质量 {m(lo, 'mass_kg'):.1f} → "
            f"{m(hi, 'mass_kg'):.1f} kg，膝峰值需求 {m(lo, 'knee_tau_Nm'):.0f} → {m(hi, 'knee_tau_Nm'):.0f} Nm，"
            f"步行 CoT {m(lo, 'cot_walk'):.2f} → {m(hi, 'cot_walk'):.2f}。</li>")
    if not out:
        return ""
    return ("<h2>参数怎样影响结果</h2><ul>" + "".join(out) + "</ul>"
            '<p class="sub">腿越长，步长越大、步行能耗越低，但同样屈膝下膝关节力臂更长、扭矩更大、整机更重；'
            '这是腿长比要在质量/成本与能耗/速度之间权衡的原因。</p>')


def _population_many(pts: list) -> str:
    plt = _plt()
    from ..trends import link_fits
    from .reference import Z_SPAN_SHARE
    refs = load_reference()
    mv = link_fits("humanoid").mass_vs_height
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter([r.height_est_m for r in refs], [r.mass_kg for r in refs], s=22, color=MUTED,
               label="published humanoids (survey)")
    hh = np.linspace(0.8, 2.1, 100)
    ax.plot(hh, mv["coef"] * (hh * Z_SPAN_SHARE) ** mv["exp"], color=MUTED, lw=1.2, ls="--",
            label="Draft mass-vs-size trend")
    ax.scatter([p["height_m"] for p in pts], [p["mass_kg"] for p in pts], s=26, color=SERIES[0],
               label="screened concepts", zorder=3)
    ax.set_xlabel("standing height (m)")
    ax.set_ylabel("total mass (kg)")
    ax.legend(fontsize=8)
    return _png(fig)


def benchmark_report(rows: list) -> str:
    b = ["<h1>对标：用已发布人形机器人检验粗筛模型</h1>",
         '<p class="sub">对每台机器人，按其关节跨度推算身高、按腿长算腿长比，用默认工况闭合质量环，'
         '再与厂商公布的总质量和 URDF 关节扭矩上限比较。比值约 1 说明模型数量级可信；'
         '扭矩需求未乘安全系数，厂商上限通常高于需求。</p>']
    ok = [r for r in rows if r["closed"]]
    if ok:
        mr = np.array([r["mass_ratio"] for r in ok])
        kr = np.array([r["knee_ratio"] for r in ok if r.get("knee_ratio")])
        b.append('<div class="tiles">' + "".join(f"<div class='tile'><b>{v}</b><span>{k}</span></div>" for v, k in [
            (f"{len(ok)}", "对标机型"),
            (f"{np.exp(np.mean(np.log(mr))):.2f}×", "质量预测 / 实际（几何平均）"),
            (f"{np.exp(np.mean(np.abs(np.log(mr)))):.2f}×", "质量误差倍数（几何平均）"),
            (f"{np.median(kr):.2f}" if kr.size else "–", "膝扭矩需求 / 厂商上限（中位数）")]) + "</div>")
        b.append(f'<img alt="benchmark" src="{chart_benchmark(rows)}">')
    head = ["机器人", "身高估计 m", "腿长比", "实际 kg", "预测 kg", "比值",
            "膝 实际/需求", "髋俯仰 实际/需求", "踝俯仰 实际/需求"]
    t = []
    for r in rows:
        t.append([_e(r["robot"]), f"{r['height_est_m']:.2f}", f"{r['leg_ratio']:.2f}",
                  f"{r['mass_real_kg']:.1f}", _f(r["mass_pred_kg"]), _f(r["mass_ratio"], 2)] +
                 [f"{_f(r.get(c + '_real_Nm'), 0)} / {_f(r.get(c + '_pred_Nm'), 0)}"
                  for c in ("knee", "hip_pitch", "ankle_pitch")])
    b.append(_table(head, t))
    b.append('<p class="sub">身高由关节跨度按默认比例反推，部分机型跨度包含颈部关节会高估身高；'
             '厂商 URDF 上限有的是峰值扭矩、有的是软件限位。</p>')
    return _page("粗筛模型对标", "".join(b))


def write(path: Path, html_text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path
