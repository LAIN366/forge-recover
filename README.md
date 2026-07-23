本仓库面向 Cruzr S2 双臂机器人操作研究，基于 MuJoCo 构建多模态感知、
信念任务图、分层运动规划、主动故障诊断与自主恢复闭环，并预留真实机器人接入边界。

## 仓库结构

```text
forge-recover/
├── datasets/       # 可复用的示教数据，生成的训练/实验结果不纳入版本控制
└── dual_recovery/  # 源码、配置、场景、测试、文档和机器人模型
```

完整的研究内容、算法能力、实验设计和实机接入说明见
[dual_recovery/README.md](dual_recovery/README.md)。

## 核心研究内容

1. 融合 RGB-D 6D 位姿、接触力、夹爪状态和关节状态，构建可靠性自适应的多模态信念状态。
2. 构建支持在线更新、故障回退、双臂角色切换和任务续接的协作任务图。
3. 研究任务层、双臂协调层和单臂运动层组成的分层规划，并引入恢复代价与 CVaR 风险评价。
4. 实现运输滑落、视觉退化、单臂失效、同步异常和动态障碍下的主动诊断与自主恢复。

## 快速验收

```bash
cd dual_recovery
python3 scripts/dual_arm_transport_demo.py
```

验证运输中随机滑落、视觉搜索、双臂重抓和任务续接：

```bash
cd dual_recovery
python3 scripts/dual_arm_transport_demo.py \
  --fault transport_slip --policy ours
```

运行测试：

```bash
cd dual_recovery
PYTHONPATH=scripts python3 -m pytest -q tests
```

## 研究文档

- [系统架构](dual_recovery/docs/architecture.md)
- [双臂研究协议](dual_recovery/docs/dual_arm_research_protocol.md)
- [实验协议](dual_recovery/docs/experiment_protocol.md)
- [论文执行计划](dual_recovery/docs/thesis_execution_plan.md)

训练权重、批量实验日志、统计结果、缓存和本地工具链均由 `.gitignore` 排除；
仓库只保存可复现实验所需的源码、配置、场景、模型描述和数据集。

## 通用上传工具

仓库提供不修改源目录的通用 GitHub 发布工具，支持上传预览、`.gitignore`、
自定义排除规则、Git LFS、可选建库、显式强制覆盖和远端 SHA 校验：

```bash
python3 tools/easy_publish.py /path/to/project owner/repository --dry-run
```

详细用法见 [tools/README.md](tools/README.md)。
