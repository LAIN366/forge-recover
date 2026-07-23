# 多模态感知下任务图驱动的双臂协作分层规划与故障恢复研究

本项目面向硕士论文题目：

> **多模态感知下任务图驱动的双臂协作分层规划与故障恢复研究**

基于 Cruzr S2 与 MuJoCo，研究复杂操作环境中的多模态状态感知、信念任务图、
双臂分层运动规划以及操作故障自主恢复。项目目标不是只完成仿真动作，而是形成
可比较、可消融、可统计并能够迁移到真实机器人的研究方法。

## 研究内容

1. 融合 RGB-D 6D 位姿、双臂接触力、夹爪接触和关节跟踪状态，构建带置信度的
   多模态操作信念状态。
2. 构建支持节点在线更新、故障回退、双臂角色切换和任务续接的协作任务图。
3. 研究任务层、双臂协调层和单臂运动层组成的分层规划，并以期望恢复代价和
   CVaR 尾部风险评价候选方案。
4. 研究运输滑落、视觉退化、单臂失效、同步异常和动态障碍下的主动诊断、局部
   重规划与操作恢复。

## 当前能力

- Cruzr S2 URDF/STL、双臂和动力学双指夹爪仿真
- 腰部/头部 RGB-D 相机、视觉检测框、置信度与 6D 位姿显示
- 可选 YOLO 实例分割掩膜与深度/PCA 6D 位姿恢复接口
- 视觉、深度、接触、力和关节状态的可靠性自适应融合
- 阶段条件时序贝叶斯故障后验与期望信息增益主动探测
- 上下文恢复经验图、Beta 成功率更新与经验代价 CVaR 策略选择
- 双臂阻尼最小二乘 IK、RRT-Connect 碰撞规划和同步轨迹
- 信念任务图、在线角色更新、局部回退和任务续接
- 恢复感知期望代价、最小间隙约束和 CVaR 风险评价
- 可复现的接触丢失、运输滑落、视觉遮挡、目标偏移、传感器丢包、
  单臂失效、同步延迟和动态障碍故障
- 运输滑落后的停止、腰部相机低头搜索、视觉重定位、双臂重抓、重新抬升和续运
- 重抓失败重试、单侧失联支撑臂放回和双侧再次掉落恢复
- JSONL 追踪、四组基线、配对随机种子、CSV 聚合和论文实验清单
- Cruzr S2 实机控制端口与安全门控接口

当前算法创新主线不是单独堆叠检测、任务图和恢复模块，而是建立统一闭环：任务
阶段提供故障转移先验，多模态观测递推故障后验，后验熵驱动信息增益探测，恢复
规划再联合期望代价、上下文经验成功率与 CVaR 尾部风险选择方案。每次更新记录完整后验、
熵和探测信息增益，可用于 B0/B1/B2/Ours 消融及诊断校准评价。

需要跨回合积累恢复经验时显式指定文件：

```bash
python3 scripts/dual_arm_transport_demo.py \
  --fault transport_slip --policy ours \
  --experience-graph results/train/recovery_experience.json
```

正式实验必须先用训练种子生成经验图，再冻结后用于验证和测试；禁止在测试种子上
更新后再报告同一批结果。

未见工件参数域验证：

```bash
python3 scripts/dual_arm_transport_demo.py \
  --workpiece-domain unseen --seed 7 --policy ours
```

训练好的工件分割权重接入：

```bash
python3 scripts/dual_arm_transport_demo.py \
  --workpiece-domain unseen --detector yolo-seg \
  --detector-weights models/workpiece_yolo11s_seg.pt
```

RTX 3050 Laptop 4 GiB 建议训练 `YOLO11s-seg`，使用 640 输入、AMP 和 batch 2～4；
远端虚拟机没有 GPU，只部署导出的 ONNX/OpenVINO 模型。当前仅承诺工作空间与负载
范围内的刚性可抓取工件，透明物、软体、超载、无可行抓取面等情况必须安全拒绝，
不能宣称对任意物体无条件成功。

当前未见参数域已通过 seeds `7、17、27` 的正常运输验证，并通过 seed `7` 的
“未见工件参数 + 随机运输滑落”组合恢复验证。抓取跨度、抓取高度和左右目标均由
检测姿态与工件几何自适应计算，不再固定沿世界坐标轴布置。

## 目录结构

```text
dual_recovery/
├── configs/                         # 可复现实验配置
├── docs/                            # 架构、实验协议和论文执行计划
├── robots/                          # Cruzr S2 URDF 与网格
├── scripts/
│   ├── cruzr_sim/
│   │   ├── adapters/                # ROS 2 与真实机器人适配边界
│   │   │   └── cruzr_sdk.py         # CruzrSdkBackend 实机执行后端
│   │   ├── control/                 # 双臂 IK 与动力学夹爪
│   │   ├── diagnosis/               # 多模态异常检测与主动诊断
│   │   ├── experiments/             # 指标、统计与实验设计
│   │   ├── faults/                  # 种子化故障注入
│   │   ├── perception/              # RGB-D、传感器同步与信念融合
│   │   ├── planning/                # 分层规划、碰撞检测与风险代价
│   │   ├── recovery/                # 约束恢复与滑落恢复策略
│   │   ├── scenes/                  # 所有任务场景和场景参数
│   │   ├── simulation/
│   │   │   └── execution_backend.py # MuJoCoDualArmBackend 仿真执行后端
│   │   └── tasks/                   # 任务图和物理任务执行
│   │       └── execution_backend.py # 仿真/实机共用执行接口
│   ├── dual_arm_transport_demo.py   # 双臂协同运输主入口
│   ├── physical_grasp_demo.py       # 前期单臂抓取基线
│   ├── run_dual_arm_experiments.py  # 双臂配对批量实验
│   └── generate_research_manifest.py
└── tests/                            # 核心算法单元测试
```

所有新场景必须放在 `scripts/cruzr_sim/scenes/`。Python 任务代码只负责加载场景、
感知、诊断、规划和控制，不再内嵌大段场景 XML。

## 仿真验收

在项目根目录运行正常双臂运输：

```bash
python3 scripts/dual_arm_transport_demo.py
```

运行运输中随机滑落与自主恢复：

```bash
python3 scripts/dual_arm_transport_demo.py \
  --fault transport_slip --policy ours
```

运行动态障碍恢复：

```bash
python3 scripts/dual_arm_transport_demo.py \
  --fault dynamic_obstacle --policy ours
```

无界面回归：

```bash
MUJOCO_GL=egl PYTHONPATH=scripts python3 scripts/dual_arm_transport_demo.py \
  --headless --fault transport_slip --policy ours --seed 7 --timeout 180
```

抓取规划使用 RGB-D 检测得到的位置和姿态，不使用 MuJoCo 物体真值作为抓取依据。

## 对照方法

| 方法 | 任务模型 | 不确定性 | 主动诊断 | 恢复风险代价 | 角色切换 |
|---|---|---:|---:|---:|---:|
| B0 | 固定 FSM | 否 | 否 | 否 | 否 |
| B1 | 确定性任务图 | 否 | 否 | 否 | 否 |
| B2 | 信念任务图 | 是 | 是 | 否 | 否 |
| Ours | 信念任务图 | 是 | 是 | 是 | 是 |

机制级运输滑落先导实验中，B0/B1 最终位姿失败，B2/Ours 完成恢复。该结果只用于
验证实验链路，不能替代多种子论文统计。

## 论文级实验

生成 30 个配对种子、三档严重度和四种方法的实验清单：

```bash
PYTHONPATH=scripts python3 scripts/generate_research_manifest.py \
  --trials 30 --output results/paper/research_manifest.json
```

运行小规模先导实验：

```bash
MUJOCO_GL=egl PYTHONPATH=scripts python3 scripts/run_dual_arm_experiments.py \
  --faults transport_slip vision_occlusion synchronization_delay \
  --severities 0.5 1.0 1.5 --seeds 1000 1001 1002 \
  --output results/pilot
```

正式实验每个“方法-故障-严重度”单元至少运行 30 个相同配对种子，报告成功率、
恢复率、故障分类准确率、规划与执行时间、同步误差、最小间隙、回退深度、重规划
次数、95% 置信区间、效应量和 CVaR。训练、验证和测试种子必须分离。

## 测试

```bash
cd ~/robosuite_ws/dual_recovery
PYTHONPATH=scripts python3 -m pytest -q tests
```

当前完整回归：

```text
97 passed
```

运输滑落闭环已通过种子 7、17、27 的机制级物理回归，分别覆盖直接恢复、单侧
失联后支撑臂放回，以及双侧再次掉落后的视觉重试。

## 实机接入边界

真实 Cruzr S2 通过 `RobotCommandPort` 接入，任务图、诊断与规划层不直接依赖
优必选 SDK。所有关节命令必须经过 `SafetyGatedCommandPort`，检查操作员使能、
关节数量、有限值、关节限位、最大单步变化和最短执行时间。

统一的 `DualArmExecutionBackend` 为任务算法与执行设备解耦提供迁移边界：

- `scripts/cruzr_sim/tasks/execution_backend.py`：统一数据结构和执行接口。
- `scripts/cruzr_sim/simulation/execution_backend.py`：`MuJoCoDualArmBackend`，负责
  仿真观测、双臂规划、插值路点执行、夹爪和相机视角控制。
- `scripts/cruzr_sim/adapters/cruzr_sdk.py`：`CruzrSdkBackend`，负责以后封装优必选
  SDK 或 ROS 2 控制接口。

当前多模态诊断观测、腰部低头搜索和全部双臂夹爪命令已经迁移到 MuJoCo 后端，
轨迹执行仍在分阶段迁移。诊断中的物体位姿来自缓存的 RGB-D 检测结果，MuJoCo
物体真值只保留在调试元数据和物理验收中，不作为抓取与故障判定依据。
实机骨架默认禁止动作；双臂同步路点须左右两侧全部通过安全校验后才下发，避免
单侧已运动而另一侧命令被拒绝。

实机验证顺序为：只读遥测、离线回放、电机关闭联调、单臂低速空载、双臂空载、
软物体抓取，最后才启用故障恢复。完成碰撞约束、看门狗、SDK 返回码检查和独立
急停验收前，不允许自动执行恢复轨迹。

## 文档

- `docs/dual_arm_research_protocol.md`：研究假设、对照组和统计要求
- `docs/thesis_execution_plan.md`：十项工作、复现实验和实机接入计划
- `docs/architecture.md`：系统架构与模块边界
- `docs/experiment_protocol.md`：单臂前期实验协议

LLM API 不是核心方法依赖。若后续加入 LLM，只允许生成高层恢复原语候选，轨迹、
碰撞安全和执行权限仍由确定性规划器与安全层控制。
