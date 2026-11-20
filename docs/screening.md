# 人形机器人概念粗筛（screening）

在画 CAD、在跑 RL 之前，用几分钟回答这些问题：

- 这个**身高**、这个**腿长比**，整机大概多重？
- 每个关节要多大的**峰值扭矩、转速、持续扭矩**？目录里**有没有现成模组**能满足？最轻的是哪几款？
- 电池要多少 **Wh**、多重、峰值电流多大？
- 改**脚长、踝关节机构（并联/串联、电机上移）、手臂负载、续航**，结果怎么变？

它不需要 MuJoCo，也不需要 GPU，只用 numpy、scipy、pyyaml、matplotlib，笔记本 CPU 上单个方案约 5 秒，
35 个方案的扫描约 1–2 分钟。结果可以导出成 Draft 的参数覆盖文件，再用
`scripts/generate_robot.py` 生成 MJCF，进入详细仿真。

> 这是 Draft 论文之外新增的模块（`src/draft/screening/`），复用了 Draft 的拟合趋势
> （`draft.trends`）、114 款执行器目录和 29 台人形机器人调研数据，没有改动论文代码。

## 快速开始

```shell
pip install -e ".[dev]" -c constraints.txt     # 或者最小依赖：
# pip install numpy scipy pyyaml matplotlib pytest && pip install -e . --no-deps

# 1) 单个方案（趋势虚拟模组，连续定型）
python scripts/screen_humanoid.py concept --config experiments/screening/concept.yaml
# 2) 同一方案，从目录真实模组里选型
python scripts/screen_humanoid.py concept --config experiments/screening/concept.yaml --mode catalog
# 3) 临时改参数
python scripts/screen_humanoid.py concept --config experiments/screening/concept.yaml \
    --set body.height_m=1.65 --set body.leg_ratio=0.46 --set name=h165
# 4) 身高 × 腿长比扫描（1.2–1.8 m），并对每个身高的最佳组合做目录选型
python scripts/screen_humanoid.py sweep --config experiments/screening/sweep_height_leg.yaml
# 5) 用已发布的人形机器人检验模型
python scripts/screen_humanoid.py benchmark
# 测试
pytest tests/test_screening.py
```

输出都在 `generated/screening/<name>/`：

| 文件 | 内容 |
|---|---|
| `report.html` | 中文报告：概要、检查提示、关节需求与选型、扭矩–转速图、几何与质量、能耗电池、工况参数、假设 |
| `requirements.csv` | 每个执行器位的峰值扭矩/转速/功率、发热 RMS、选定模组与裕度 |
| `catalog_candidates.csv` | 每个关节前 N 个满足要求的目录模组（按质量升序） |
| `summary.json` | 全部数值与完整配置（可复现） |
| `draft_overrides.yaml` | 交给 Draft 生成 MJCF 的参数覆盖 |
| `sweep.csv` / `sweep_report.html` | 扫描表与热力图、Pareto 前沿、按身高的推荐方案 |
| `best/<点>/report.html` | 扫描中每个身高最佳方案的目录选型报告 |

## 它怎么算

```
概念参数 ──► 几何（身高 × 比例）
   │
   ▼
质量模型 ◄──────────────────────────────┐
   │  执行器（每个关节一个）             │
   │  四肢结构（Draft 人形结构趋势）     │
   │  躯干壳体（整机质量的比例）         │
   │  电池 + 计算 + 传感器 + 手           │
   ▼                                    │
工况参考动作 → 平面逆动力学 → 关节 τ(t)、ω(t) │
   ▼                                    │
每个执行器位的需求（峰值、包络、发热 RMS） │
   ▼                                    │
选执行器（趋势 / 目录）+ 电池定容 ───────┘   重复直到总质量不再变化
```

**质量螺旋**：执行器变重 → 整机变重 → 需要更大的执行器和电池。迭代收敛就是可闭合的方案；
超过 `convergence.max_mass_kg` 仍在增长，就判定为闭合不了（报告会写明）。

### 几何

站立身高 `H` 自下而上 = 踝高 + 腿长（髋—踝）+ 髋—肩 + 肩—头顶。`leg_ratio` 改变腿长时，
髋—肩随之变化，头颈比例保持不变。手臂、脚、髋宽、肩宽都按身高比例给出（`body.*`）。

### 动力学

- **矢状面**（俯仰关节）：根在支撑脚踝，脚固定在地面，平面递推牛顿–欧拉逆动力学
  （`planar.py`，已对照双摆拉格朗日方程验证）。单脚支撑模型包含摆动腿；双脚对称工况用半身模型。
  电机转子惯量按 `N²·J` 计入。
- **冠状面**（横滚关节）：步行时质心按倒立摆左右摆动，算髋横滚和踝横滚。
- **偏航、腕部、肩偏航**：准静态估算（脚底摩擦力矩、上身偏航惯量、负载力臂）。

### 工况（`tasks.*`）

| 工况 | 动作 | 主要考验 |
|---|---|---|
| `stand` | 双脚站立，膝微屈（可选双手持重） | 持续发热 |
| `walk` / `walk_fast` | 倒立摆质心 + 步长按 Froude 数缩放 + 腿部逆运动学 | 髋、膝持续发热；速度 |
| `stairs` | 上一级、下一级台阶，质心升降一个台阶高 | 膝扭矩（下楼离心） |
| `squat_lift` | 深蹲下去再起来，双手持重，躯干前倾使质心保持在脚上 | 膝、髋 |
| `stoop_lift` | 腿基本伸直、弯腰到负载再起身 | 髋俯仰、腰 |
| `run` | 弹簧–质量支撑相：正弦垂直力（由占空比定峰值），力沿腿方向 | 膝、踝峰值扭矩与功率 |
| `arm_lift` | 双臂伸直从下垂抬到水平，带负载 | 肩、肘 |
| `turn` | 原地转向 | 髋偏航、腰偏航 |
| `balance` | 单脚承重时踝把压力中心推到脚尖/脚边的能力 | 踝 |

另外每个关节有**敏捷性下限**（`actuators.speed_floor_rad_s`）：在 30% 峰值负载下也要能达到这个转速，
覆盖参考动作里没有的快速平衡恢复、侧向迈步。

### 执行器

- **扭矩–转速包络**：电流限 + 电压线，`τ(ω) = min(τ_peak, k·τ_peak·(1−ω/ω_NL))`。
  `k = actuators.voltage_limit_factor`，默认 2；`k = 1` 就是 Draft 目录的直线约定（最保守）。
- **趋势模组**（`mode: trend`）：在减速比网格上找最轻的假想模组，
  质量 `m = 0.0653·τ^0.666`，空载转速 `ω = 228.5·N^-0.938`，并检查不超过目录的
  扭矩/功率密度前沿（`frontier: p90` 或 `max`）。超出前沿会报“执行器不可行”——意思是
  现有产品做不到，需要定制或降低指标。
- **目录模组**（`mode: catalog`）：从 114 款真实模组（CubeMars、MyActuator、RobStride、Unitree、
  ZeroErr、HEBI、SteadyWin、maxon 等）以及你的自定义清单中，选覆盖所有工况采样点、
  且持续扭矩满足发热要求的最轻型号；没有满足的就退回趋势模组并标注。
- **持续扭矩**：目录没有额定值，按 `峰值 / peak_to_continuous`（QDD 3、MidGear 2.5、Harmonic 2）估算。
  自定义清单里写 `tau_cont_Nm` 就用你的值。
- **同型号分组**：`actuators.groups: {leg_big: [hip_pitch, knee]}` 让一组关节用同一型号，减少物料种类。
- **并联踝**：两电机等分俯仰与横滚（`τ_A=(τ_p+τ_r)/2`，`ω_A=ω_p+ω_r`，功率守恒），连杆比按 1。

### 能耗与电池

每个工况的平均电功率 = 正机械功 / 驱动效率 + 铜损 + 计算与传感器功耗（可设回馈效率）。
铜损用热容量假设：执行器在持续扭矩下每公斤发热 `copper_loss_W_per_kg`。按 `mission` 的时间分配加权，
乘续航时间，加储备，得到电池 Wh、质量、体积；峰值功率按同一时刻工作的关节逐点求和，
得到母线峰值电流和所需放电倍率。

## 读结果

- **状态**：可行 / 可行（有提示）/ 有执行器超出产品能力 / 无法闭合。
- **扭矩裕度、热裕度**：已经乘过安全系数之后的余量；0 表示刚好够。
- **CoP 提示**：参考动作没有利用躯干和手臂平衡，名义步行或搬运时压力中心出了脚底，
  说明脚偏短或动作偏激进。快步走和跑步的 CoP 只作参考，不计为提示。
- **扫描**：热力图看身高、腿长比怎么影响质量、膝扭矩、电池、步行能耗；★ 标出质量与步行能耗的
  Pareto 前沿。腿越长，步行能耗越低，但膝扭矩和整机质量越大——腿长比是这两者的权衡。

## 模型可信到什么程度（benchmark）

`python scripts/screen_humanoid.py benchmark` 把调研里 25 台已发布人形机器人（Unitree H1/H2、
Booster T1/K1、Tien Kung、AgiBot、LimX Oli、Kuavo、EngineAI 等）的身高与腿长比当作概念输入，
工况按 Froude 数缩放到各自尺寸，然后比较（默认配置，2026-11 运行）：

| 量 | 预测 / 实际（中位数） | 几何平均误差倍数 |
|---|---|---|
| 整机质量 | 1.04 | 1.14× |
| 膝扭矩需求 / 厂商关节上限 | 1.27 | 1.39× |
| 髋俯仰 | 0.79 | 1.52× |
| 髋横滚 | 0.86 | 1.41× |
| 踝俯仰 | 1.79 | 1.79× |
| 肩俯仰 | 0.72 | 1.63× |

整机质量和 Draft 自身的孪生误差（1.10×）同一量级，可以用来做身高、腿长、预算的取舍。
关节扭矩是数量级判断：膝和踝偏保守（主要来自跑步工况和踝平衡裕度），髋和肩偏乐观
（厂商往往给髋装与膝同级的电机，用于起身、抗推等本工具未建模的动作）。
身高由关节跨度反推，部分机型偏高；厂商的 URDF 上限有的是峰值、有的是软件限位。

## 配置参考

所有键及默认值在 `src/draft/screening/spec.py` 的 `DEFAULTS`，写错键名会直接报错。常用：

| 键 | 默认 | 作用 |
|---|---|---|
| `body.height_m` | 1.50 | 站立身高 |
| `body.leg_ratio` | 0.45 | 腿长（髋—踝）/ 身高 |
| `body.thigh_fraction` | 0.50 | 大腿 / 腿长 |
| `body.foot_length_ratio` | 0.15 | 脚长 / 身高 |
| `dof.waist`, `dof.wrist` | `[yaw]`, 1 | 腰、腕自由度 |
| `mechanism.ankle` | parallel | 并联 / 串联踝 |
| `mechanism.ankle_motor_location` | knee | 踝电机上移到小腿上端（减小摆动惯量） |
| `mechanism.knee_motor_location` | knee | 膝电机放膝处或上移到髋 |
| `mass.trunk_shell_fraction` | 0.17 | 躯干框架、外壳、线束、板卡占整机比例 |
| `payload.hands_total_kg` | 5 | 双手负载 |
| `electrical.runtime_h`, `battery_Wh_per_kg` | 2, 160 | 续航目标、电池包能量密度 |
| `mission` | 步行 60%、站立 30%、楼梯 5%、搬运 5% | 续航期间时间分配，也决定发热 RMS |
| `tasks.walk.speed / max_speed` | 1.0 / 1.5 | 步行速度（m/s） |
| `tasks.run.speed` | 2.0 | 跑步速度（m/s）；`enabled: false` 关闭 |
| `tasks.stairs.step_height_m` | 0.15 | 台阶高 |
| `balance.ankle_cop_authority` | 0.5 | 踝单独把 CoP 推到脚尖/脚边的比例（1 = 完全） |
| `actuators.mode` | trend | trend / catalog |
| `actuators.*_safety_factor` | 1.3 / 1.15 / 1.2 | 扭矩 / 转速 / 发热安全系数 |
| `actuators.voltage_limit_factor` | 2.0 | 包络形状（1 = 目录直线） |
| `actuators.extra_catalog` | – | 自定义模组 YAML/CSV，格式见 `experiments/screening/my_actuators.example.yaml` |

扫描文件就是概念 YAML 加一个 `sweep:` 块，键是点号路径，值是列表，取全组合，例如
`mechanism.ankle_motor_location: [ankle, knee]`、`payload.hands_total_kg: [3, 5, 10]`。

## 交给 Draft，以及下一步

`draft_overrides.yaml` 把身高比例、每个执行器位的电机类（峰值扭矩、空载转速、减速比；目录件用
`_<cls>_motor_catalog` 锚定）、不存在的自由度（`<role>_dof: false`）写成 Draft 人形的参数覆盖：

```shell
pip install -e ".[editor]" -c constraints.txt          # 需要 MuJoCo
python scripts/generate_robot.py --robot humanoid --overrides generated/screening/<name>/draft_overrides.yaml
python scripts/robot_tuner.py                           # 浏览器里看、继续调
```

Draft 会用自己的趋势重新算质量并跑可行性检查，数字会和粗筛略有差别；Draft 没有并联机构，
并联踝按两个串联关节导出。有 CUDA GPU 之后，再在生成的 MJCF 上训练行走策略做复核。

## 局限

- 参考动作不是控制器；平面模型不含三维耦合（如转弯时的离心、侧向推）。
- 没有建模：跌倒起身、抗推恢复、跳跃、冲击；它们往往决定髋和膝的上限，用安全系数和敏捷性下限兜底。
- 执行器质量、尺寸、转子惯量来自目录拟合（转子惯量误差约 1.56×）；持续扭矩是估算。
- 结构质量是统计趋势，不知道你的材料和工艺；`mass.structural_scale` 可以整体修正。
- 这是概念阶段的数量级工具，不能代替详细设计、多体仿真和样机测试。
