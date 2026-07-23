"""Belief-aware task graph for cooperative dual-arm manipulation."""

from dataclasses import dataclass, field
from enum import Enum


class NodeState(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ArmRole(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    DUAL = "dual"


@dataclass
class TaskNode:
    node_id: str
    dependencies: tuple[str, ...] = ()
    role: ArmRole = ArmRole.NONE
    minimum_belief: float = 0.65
    recovery_node: str | None = None
    enabled: bool = True
    state: NodeState = NodeState.BLOCKED
    belief: float = 1.0
    metadata: dict = field(default_factory=dict)


class CooperativeTaskGraph:
    """Executes a DAG and supports local rollback after execution failures."""

    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("task node ids must be unique")
        self.history = []
        self._validate_dependencies()
        self.refresh()

    def _validate_dependencies(self):
        for node in self.nodes.values():
            missing = set(node.dependencies) - self.nodes.keys()
            if missing:
                raise ValueError(f"{node.node_id} has missing dependencies: {sorted(missing)}")
            if node.recovery_node is not None and node.recovery_node not in self.nodes:
                raise ValueError(
                    f"{node.node_id} has missing recovery node: {node.recovery_node}"
                )
        visiting = set()
        visited = set()

        def visit(node_id):
            if node_id in visiting:
                raise ValueError("task dependencies must form a DAG")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in self.nodes[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def refresh(self):
        for node in self.nodes.values():
            if node.state in {NodeState.RUNNING, NodeState.SUCCEEDED, NodeState.FAILED}:
                continue
            dependencies_met = all(
                self.nodes[item].state == NodeState.SUCCEEDED
                for item in node.dependencies
            )
            node.state = (
                NodeState.READY
                if node.enabled
                and dependencies_met
                and node.belief >= node.minimum_belief
                else NodeState.BLOCKED
            )

    def ready_nodes(self):
        self.refresh()
        return [node for node in self.nodes.values() if node.state == NodeState.READY]

    def update_belief(self, node_id, belief):
        node = self.nodes[node_id]
        node.belief = min(1.0, max(0.0, float(belief)))
        self.history.append(("belief", node_id, node.belief))
        self.refresh()

    def update_role(self, node_id, role, rationale="online capability update"):
        node = self.nodes[node_id]
        role = role if isinstance(role, ArmRole) else ArmRole(role)
        previous = node.role
        node.role = role
        node.metadata["role_update_rationale"] = str(rationale)
        node.metadata["previous_role"] = previous.value
        self.history.append(("role", node_id, previous.value, role.value))

    def rollback_depth(self, node_id):
        """Count downstream nodes invalidated by a local graph rollback."""
        affected = {node_id}
        changed = True
        while changed:
            changed = False
            for node in self.nodes.values():
                if node.node_id not in affected and affected.intersection(
                    node.dependencies
                ):
                    affected.add(node.node_id)
                    changed = True
        return len(affected) - 1

    def start(self, node_id):
        self.refresh()
        node = self.nodes[node_id]
        if node.state != NodeState.READY:
            raise RuntimeError(f"task node is not ready: {node_id}")
        node.state = NodeState.RUNNING
        self.history.append(("start", node_id))

    def complete(self, node_id):
        node = self.nodes[node_id]
        if node.state != NodeState.RUNNING:
            raise RuntimeError(f"task node is not running: {node_id}")
        node.state = NodeState.SUCCEEDED
        self.history.append(("complete", node_id))
        self.refresh()

    def fail(self, node_id, reason, recovery_node_id=None):
        node = self.nodes[node_id]
        if node.state != NodeState.RUNNING:
            raise RuntimeError(f"task node is not running: {node_id}")
        node.state = NodeState.FAILED
        node.metadata["failure_reason"] = str(reason)
        self.history.append(("fail", node_id, str(reason)))
        selected_recovery = recovery_node_id or node.recovery_node
        if selected_recovery is None:
            self.refresh()
            return None
        if selected_recovery not in self.nodes:
            raise ValueError(f"unknown recovery node: {selected_recovery}")
        recovery = self.nodes[selected_recovery]
        recovery.enabled = True
        recovery.state = NodeState.READY
        recovery.metadata["failed_node"] = node_id
        recovery.metadata["failure_reason"] = str(reason)
        return recovery

    def reset_from(self, node_id):
        """Invalidate a node and all downstream work before local replanning."""
        affected = {node_id}
        changed = True
        while changed:
            changed = False
            for node in self.nodes.values():
                if node.node_id not in affected and affected.intersection(node.dependencies):
                    affected.add(node.node_id)
                    changed = True
        for affected_id in affected:
            self.nodes[affected_id].state = NodeState.BLOCKED
        self.history.append(("reset", node_id, tuple(sorted(affected))))
        self.refresh()

    def resume_after_recovery(self, recovery_node_id, failed_node_id):
        """Commit a recovery node and resume the interrupted transaction."""
        recovery = self.nodes[recovery_node_id]
        failed = self.nodes[failed_node_id]
        if recovery.state != NodeState.RUNNING:
            raise RuntimeError("recovery node is not running")
        if failed.state != NodeState.FAILED:
            raise RuntimeError("interrupted node is not failed")
        recovery.state = NodeState.SUCCEEDED
        recovery.enabled = False
        failed.state = NodeState.RUNNING
        failed.metadata["recovery_count"] = int(
            failed.metadata.get("recovery_count", 0)
        ) + 1
        self.history.append((
            "resume", recovery_node_id, failed_node_id,
            failed.metadata["recovery_count"],
        ))
        self.refresh()


def build_cooperative_transport_graph():
    """Build the main dual-arm transport task with active recovery branches."""
    return CooperativeTaskGraph([
        TaskNode("observe_scene", minimum_belief=0.55),
        TaskNode("estimate_object_pose", ("observe_scene",), minimum_belief=0.70,
                 recovery_node="active_reobserve"),
        TaskNode("assign_arm_roles", ("estimate_object_pose",)),
        TaskNode("plan_dual_pregrasp", ("assign_arm_roles",), role=ArmRole.DUAL,
                 recovery_node="reassign_roles"),
        TaskNode("dual_grasp", ("plan_dual_pregrasp",), role=ArmRole.DUAL,
                 recovery_node="reassign_roles"),
        TaskNode("verify_grasp", ("dual_grasp",), role=ArmRole.DUAL,
                 recovery_node="active_reobserve"),
        TaskNode("cooperative_lift", ("verify_grasp",), role=ArmRole.DUAL,
                 recovery_node="stabilize_object"),
        TaskNode("cooperative_transport", ("cooperative_lift",), role=ArmRole.DUAL,
                 recovery_node="stabilize_object"),
        TaskNode("place_object", ("cooperative_transport",), role=ArmRole.DUAL,
                 recovery_node="reassign_roles"),
        TaskNode("retreat_arms", ("place_object",), role=ArmRole.DUAL,
                 recovery_node="reassign_roles"),
        TaskNode("verify_completion", ("retreat_arms",), minimum_belief=0.75,
                 recovery_node="active_reobserve"),
        TaskNode("active_reobserve", minimum_belief=0.0, enabled=False),
        TaskNode("reassign_roles", minimum_belief=0.0, enabled=False),
        TaskNode(
            "stabilize_object", role=ArmRole.DUAL,
            minimum_belief=0.0, enabled=False,
        ),
    ])
