"""Optional ROS 2 runtime for Cruzr S2 diagnosis and recovery-plan output."""

import json


def run_ros2_node():
    """Start the node inside the Cruzr S2 ROS 2 Humble container.

    Imports are intentionally local so offline tests do not require UBTECH's
    custom message packages.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        from mc_state_msgs.msg import RobotState
        from ecat_task_msgs.msg import GripStatus
    except ImportError as error:
        raise RuntimeError(
            "Cruzr ROS 2 message packages are required; run inside the robot "
            "Ubuntu 22.04 Humble developer container"
        ) from error

    from cruzr_sim.adapters.cruzr_ros2 import (
        CruzrObservationAssembler,
        CruzrTelemetry,
        CruzrTopicMap,
    )
    from cruzr_sim.tasks.manipulation_supervisor import ManipulationSupervisor

    class CruzrDiagnosisNode(Node):
        def __init__(self):
            super().__init__("cruzr_active_diagnosis")
            topics = CruzrTopicMap()
            self.assembler = CruzrObservationAssembler()
            self.supervisor = ManipulationSupervisor()
            self.robot_state = None
            self.gripper_state = None
            self.object_pose = None
            self.tool_pose = None
            self.stage = "idle"
            self.previous_object_z = None
            self.previous_time = None
            self.last_plan_id = None

            sensor_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                RobotState, topics.robot_state,
                self._set_robot_state, sensor_qos,
            )
            self.create_subscription(
                GripStatus, topics.right_gripper_state,
                self._set_gripper_state, sensor_qos,
            )
            self.create_subscription(
                PoseStamped, "/diagnosis/object_pose",
                self._set_object_pose, sensor_qos,
            )
            self.create_subscription(
                PoseStamped, "/diagnosis/tool_pose",
                self._set_tool_pose, sensor_qos,
            )
            self.create_subscription(
                String, "/task/stage", self._set_stage, 10,
            )
            self.diagnosis_publisher = self.create_publisher(
                String, "/diagnosis/report", 10
            )
            self.probe_publisher = self.create_publisher(
                String, "/diagnosis/probe_request", 10
            )
            self.plan_publisher = self.create_publisher(
                String, "/recovery/plan", 10
            )
            self.create_timer(0.1, self._tick)
            self.get_logger().info("Cruzr active diagnosis node started at 10 Hz")

        def _set_robot_state(self, message):
            self.robot_state = message

        def _set_gripper_state(self, message):
            self.gripper_state = message

        def _set_object_pose(self, message):
            self.object_pose = message

        def _set_tool_pose(self, message):
            self.tool_pose = message

        def _set_stage(self, message):
            self.stage = message.data

        @staticmethod
        def _position(message):
            point = message.pose.position
            return float(point.x), float(point.y), float(point.z)

        def _right_wrench(self):
            for item in self.robot_state.ft_states:
                frame = item.header.frame_id.lower()
                if "right" in frame or frame.startswith("r_"):
                    force = item.wrench.force
                    torque = item.wrench.torque
                    return (
                        (float(force.x), float(force.y), float(force.z)),
                        (float(torque.x), float(torque.y), float(torque.z)),
                    )
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        def _object_vertical_velocity(self, now, object_position):
            velocity = 0.0
            if self.previous_time is not None and now > self.previous_time:
                velocity = (
                    object_position[2] - self.previous_object_z
                ) / (now - self.previous_time)
            self.previous_time = now
            self.previous_object_z = object_position[2]
            return velocity

        def _tick(self):
            if any(item is None for item in (
                self.robot_state,
                self.gripper_state,
                self.object_pose,
                self.tool_pose,
            )):
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            object_position = self._position(self.object_pose)
            force, torque = self._right_wrench()
            joint_state = self.robot_state.joint_states
            gripping = int(self.gripper_state.grip_state) == 2
            telemetry = CruzrTelemetry(
                timestamp=now,
                stage=self.stage,
                object_position=object_position,
                tool_position=self._position(self.tool_pose),
                joint_position=tuple(float(value) for value in joint_state.position),
                joint_velocity=tuple(float(value) for value in joint_state.velocity),
                wrench_force=force,
                wrench_torque=torque,
                left_contact=gripping,
                right_contact=gripping,
                gripper_current=float(self.gripper_state.cur),
                object_vertical_velocity=self._object_vertical_velocity(
                    now, object_position
                ),
            )
            decision = self.supervisor.observe(self.assembler.build(telemetry))
            if decision.report.anomalous:
                self._publish_json(
                    self.diagnosis_publisher, decision.report.to_dict()
                )
            if decision.probe:
                self._publish_json(self.probe_publisher, {
                    "probe": decision.probe,
                    "suspected_fault": decision.confirmed_fault.value,
                    "confidence": decision.confidence,
                })
            plan = decision.recovery_plan
            if plan is not None and plan.plan_id != self.last_plan_id:
                self.last_plan_id = plan.plan_id
                self._publish_json(self.plan_publisher, plan.to_dict())

        @staticmethod
        def _publish_json(publisher, payload):
            message = String()
            message.data = json.dumps(payload, ensure_ascii=False)
            publisher.publish(message)

    rclpy.init()
    node = CruzrDiagnosisNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
