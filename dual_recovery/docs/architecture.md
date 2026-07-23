# 系统架构

## 设计原则

1. 核心诊断与恢复算法不依赖 MuJoCo，未来可直接接 ROS 2 数据。
2. 感知、诊断、恢复规划和物理执行之间只交换类型化数据。
3. LLM/VLM 只能提出高层候选动作，不能直接输出或执行关节控制量。
4. 所有恢复计划必须通过确定性的动作、参数、顺序和安全约束检查。
5. 所有故障、诊断、主动探测和恢复结果必须写入可追溯实验日志。

## 数据流

```text
MuJoCo / Cruzr S2 ROS 2
    -> synchronized SensorPacket
    -> RGB-D detection and VisualPoseDetection
    -> ManipulationObservation
    -> StageAwareAnomalyDetector
    -> ActiveDiagnoser
    -> information-gathering probe
    -> RecoveryCaseLibrary / optional LLM candidate
    -> RecoveryConstraintChecker
    -> RecoveryPlan
    -> task-specific execution adapter
    -> RecoveryVerifier
```

## 诊断层

`ManipulationObservation` 是仿真和实机共用的数据契约，包含任务阶段、物体与
末端位姿、左右接触、接触力、物体速度、跟踪误差、IK/规划/碰撞状态及传感器
有效性。

检测器首先处理碰撞、IK 失败等即时安全事件，再利用短时间窗口判断漏抓、滑落、
不稳定抓取和目标移动。主动诊断器对连续证据进行平滑，并在置信度不足时请求
低风险探测动作。

## 恢复层

恢复计划由有限动作原语构成。案例库提供可解释基线；可选 LLM 适配器只接受严格
JSON，输出仍需经过 `RecoveryConstraintChecker`。仿真任务通过适配器把高层计划
映射为非阻塞状态机动作。

动态障碍以 MuJoCo `mocap` 刚体表示，在原轨迹执行期间进入未来路点。注入器只接受
同时满足当前构型、撤退构型和目标构型安全，且确实阻断至少一个未来路点的位置。
碰撞诊断触发后，执行器先沿远离障碍方向局部撤退，再依次验证多个绕行目标，最终
由 RRT-Connect 生成满足机械臂和夹爪三轴包络约束的新路径。

## 实机迁移边界

```text
任务图 + 故障诊断与恢复
          |
          v
DualArmExecutionBackend
     /               \
MuJoCo 后端       Cruzr SDK 后端
                       |
             SafetyGatedCommandPort
```

`DualArmExecutionBackend` 定义统一的观测、双臂位姿规划、同步路点执行、夹爪、
相机视角和停止接口。新增任务与后续迁移后的任务图只操作该协议中的 6D 位姿、双臂
遥测和执行结果，不导入 MuJoCo 或厂商 SDK。`MuJoCoDualArmBackend` 已实现感知
观测、双臂规划、按时长插值执行、夹爪和搜索视角接口，现有主循环已先迁移腰部
搜索视角，其余执行阶段将逐段迁入。当前 `CruzrSdkBackend` 是默认关闭的实机骨架，只有操作员明确
使能后才接受动作；同步路点在任一机械臂越限时会在下发前整体拒绝并触发停止。

实机只需新增 ROS 2 适配器，将以下话题转换为 `SensorPacket`：

- `/mc/sdk/robot_state`：关节、IMU、六维力
- `/ecat/right_grip/state`：夹爪位置、速度、电流和状态
- RGB-D/双目相机话题：物体位姿与视觉验证
- `/mc/sdk/robot_command`、夹爪命令与底盘速度接口：动作执行

诊断、案例检索、约束检查和实验指标代码无需修改。

`run_cruzr_diagnosis_node.py` 已实现上述订阅与发布边界。物体和末端位姿由外部视觉
或 TF 节点以 `PoseStamped` 提供；恢复计划只发布，不在诊断节点内直接执行。
