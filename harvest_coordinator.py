"""GPU PC 1용 수확 supervisor와 RobotMotion Action Client.

한 로봇의 수확 전체 주기를 관리한다. 개인 PC 1이 발행한 target을 받아
세대·시간·frame을 검증하고, RobotMotion Goal을 단계 순서대로 보낸다.

멀티로봇 운용에서는 로봇마다 하나씩 띄운다. target과 Action은 로봇
namespace 아래에 있으므로 두 supervisor가 서로의 로봇을 움직이지 않는다.
다만 컨베이어는 한 대뿐이므로, 배치 구간만 ``/conveyor/place_command``
mutex로 직렬화한다. 이 락이 없으면 두 로봇이 같은 순간에 같은 벨트 위로
팔을 뻗는다.

실행:
    source /opt/ros/jazzy/setup.bash
    ROS_DOMAIN_ID=102 python3 harvest_coordinator.py --robot-id robot_01
    ROS_DOMAIN_ID=102 python3 harvest_coordinator.py --robot-id robot_01 --execute

``--execute`` 없이 띄우면 target 수신과 검증만 관측하고 로봇에 Goal을
보내지 않는다. 배선을 먼저 확인할 때 쓴다.
"""

import argparse
import math
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from appleproj_interfaces.action import RobotMotion
from appleproj_interfaces.msg import (
    HarvestPerceptionStatus,
    HarvestTarget,
    MotionStatus,
    PlaceCoordinatorStatus,
    PlanningScene,
    SimulationState,
)
from appleproj_interfaces.srv import GetPlanningScene, PlaceCommand
from geometry_msgs.msg import PoseStamped

from harvest_namespace import HarvestNames, add_robot_id_argument
from harvest_route_planner import (
    RoutePlanningError,
    approach_orientation_xyzw,
    validate_scene_version,
)


# ══════════════════════════════════════════════════════════════
# 수확 단계 순서 (docs/features/harvesting.md)
# ══════════════════════════════════════════════════════════════
SEQUENCE = (
    RobotMotion.Goal.APPROACH,
    RobotMotion.Goal.GRASP,
    RobotMotion.Goal.TWIST,
    RobotMotion.Goal.PULL,
    RobotMotion.Goal.TRANSPORT,
    RobotMotion.Goal.PLACE,
    RobotMotion.Goal.RELEASE,
    RobotMotion.Goal.RETRACT,
)

MOTION_NAMES = {
    RobotMotion.Goal.APPROACH: "APPROACH",
    RobotMotion.Goal.GRASP: "GRASP",
    RobotMotion.Goal.TWIST: "TWIST",
    RobotMotion.Goal.PULL: "PULL",
    RobotMotion.Goal.TRANSPORT: "TRANSPORT",
    RobotMotion.Goal.PLACE: "PLACE",
    RobotMotion.Goal.RELEASE: "RELEASE",
    RobotMotion.Goal.RETRACT: "RETRACT",
}

# 접촉 이후 단계. 여기서 실패하면 사과를 들었거나 로봇이 나무 안에 있을 수
# 있으므로 다음 Goal을 보내지 않고 안전 정지한다.
POST_CONTACT_MOTIONS = frozenset(
    {
        RobotMotion.Goal.GRASP,
        RobotMotion.Goal.TWIST,
        RobotMotion.Goal.PULL,
        RobotMotion.Goal.TRANSPORT,
        RobotMotion.Goal.PLACE,
        RobotMotion.Goal.RELEASE,
    }
)

# 컨베이어 락이 필요한 구간. TRANSPORT 직전에 잡고 RELEASE 뒤에 놓는다.
PLACE_LOCK_START = RobotMotion.Goal.TRANSPORT
PLACE_LOCK_END = RobotMotion.Goal.RELEASE

# 로봇 base pose를 조회할 TF frame이다.
#
# Isaac의 ROS2PublishTransformTree에는 frame prefix 입력이 없어서, 두 로봇의
# 링크를 모두 발행하면 양쪽 다 base_link/link_1/palm 이라는 같은 이름을
# 주장한다. 그래서 로봇별로 유일한 USD 루트 prim 이름을 base frame으로 쓴다.
# 이 이름은 USD에서 그대로 온 것이고 새로 지어낸 값이 아니다.
BASE_FRAME_BY_ROBOT = {
    "robot_01": "m0617_01",
    "robot_02": "m0617_02",
}

# 물리 수확 TCP는 USD palm frame에서 palm 로컬 +Y로 0.0908 m 떨어진 점이다.
# 멀티로봇에서는 위 이유로 palm frame을 TF에서 구분할 수 없다. GRASP와
# RELEASE는 "현재 pose를 유지하고 그리퍼만 여닫는" 단계라 GPU PC 1의 Action
# 서버가 실제 현재 pose를 그대로 쓰므로, 이 좌표가 실행 목표를 바꾸지 않는다.
TCP_FRAME = "palm"
PALM_TO_TCP_Y_M = 0.0908

# Threshold 값은 통합 시험 전까지 TBD다. 음수 sentinel은 해당 검사를
# 비활성화하며, 범위·세대·frame·timestamp 일치 검사는 항상 수행한다.
TARGET_MAX_AGE_SEC = -1.0
MIN_TARGET_CONFIDENCE = -1.0
MIN_VALID_DEPTH_RATIO = -1.0
MAX_TF_TIME_ERROR_SEC = -1.0

# 한 프레임에서 여러 target이 연달아 오므로, 배치가 다 도착할 때까지
# 잠깐 모았다가 거리순으로 정렬한다.
TARGET_BATCH_DEBOUNCE_SEC = 0.05

# 사과를 쥔 뒤에 place를 요청하므로, 서비스 탐색이 늦었다는 이유만으로
# 즉시 포기하면 안 된다.
PLACE_SERVICE_WAIT_SEC = 5.0

APPLE_IDS_BY_ROBOT = {
    "robot_01": ("apple_001", "apple_002", "apple_003"),
    "robot_02": ("apple_004", "apple_005", "apple_006"),
}
PLACE_IDS_BY_ROBOT = {
    "robot_01": "CONVEYOR_PLACE_01_LANDING",
    "robot_02": "CONVEYOR_PLACE_02_LANDING",
}

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def optional_threshold(value):
    """음수 sentinel이면 검사하지 않는다는 뜻이다."""
    value = float(value)
    return None if value < 0.0 else value


def assign_apple_ids_by_distance(robot_id, robot_position, candidates):
    """카메라 로컬 target ID를 승인된 전역 apple ID로 고정한다."""
    if robot_id not in APPLE_IDS_BY_ROBOT:
        raise ValueError(f"지원하지 않는 robot_id입니다: {robot_id}")
    robot_position = np.asarray(robot_position, dtype=float)
    if robot_position.shape != (3,) or not np.all(np.isfinite(robot_position)):
        raise ValueError("robot position은 유효한 world XYZ여야 합니다.")
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            float(np.linalg.norm(np.asarray(candidate.center) - robot_position)),
            str(candidate.key),
        ),
    )
    ids = APPLE_IDS_BY_ROBOT[robot_id]
    if len(ordered) > len(ids):
        raise ValueError(f"{robot_id}은 최대 {len(ids)}개 사과만 ID를 부여할 수 있습니다.")
    return {candidate.key: ids[index] for index, candidate in enumerate(ordered)}


@dataclass
class PendingTarget:
    """아직 시작하지 않았거나 실행 중인 수확 대상 하나."""

    key: tuple
    message: HarvestTarget
    center: np.ndarray
    stamp_ns: int
    apple_id: str = ""
    attempts: int = 0


class HarvestCoordinator(Node):
    """target 대기열과 단계 순서를 관리하는 supervisor."""

    def __init__(self, robot_id, execute):
        self.names = HarvestNames(robot_id)
        super().__init__(f"harvest_coordinator_{self.names.robot_id}")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.execute = bool(execute)
        self.robot_id = self.names.robot_id
        self.place_position_id = PLACE_IDS_BY_ROBOT[self.robot_id]

        # 시뮬레이션 세대
        self.simulation_state = None
        self.reset_id = 0
        self.scene_version = 0
        self.scene = None

        # 대기열
        self.pending = {}
        self.retry_queue = deque()
        self.completed = set()
        self.failed = set()
        self.active = None
        self.sequence_index = 0
        self.goal_handle = None
        self.generation = 0
        self.safety_stopped = False
        self.batch_timer = None

        # 컨베이어 락
        self.place_state = None
        self.reservation_id = ""
        self.place_locked = False
        self.waiting_for_place_lock = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            HarvestTarget, self.names.target_topic, self.on_target, RELIABLE_QOS
        )
        self.create_subscription(
            HarvestPerceptionStatus,
            self.names.perception_status_topic,
            self.on_perception_status,
            RELIABLE_QOS,
        )
        self.create_subscription(
            SimulationState,
            self.names.simulation_state_topic,
            self.on_simulation_state,
            LATCHED_QOS,
        )
        self.create_subscription(
            PlanningScene, self.names.planning_scene_topic, self.on_scene, LATCHED_QOS
        )
        self.create_subscription(
            MotionStatus,
            self.names.motion_status_topic,
            self.on_motion_status,
            RELIABLE_QOS,
        )
        self.create_subscription(
            PlaceCoordinatorStatus,
            self.names.conveyor_place_status_topic,
            self.on_place_status,
            RELIABLE_QOS,
        )

        # 배치 락 서비스 응답 콜백 안에서 Action Goal 을 보내고, 그 Goal 의
        # 응답과 결과 콜백을 다시 받아야 한다. 기본 MutuallyExclusive 그룹에
        # 두면 바깥 콜백이 그룹을 잡고 있는 동안 안쪽 콜백이 실행될 자리가
        # 없어서, 서버가 TRANSPORT 를 끝냈는데도 결과가 영영 도착하지 않는다.
        # 실측에서 조율기가 "-> TRANSPORT" 에서 멈춰 있었다.
        self.async_group = ReentrantCallbackGroup()
        self.action_client = ActionClient(
            self,
            RobotMotion,
            self.names.robot_motion_action,
            callback_group=self.async_group,
        )
        self.scene_client = self.create_client(
            GetPlanningScene,
            self.names.planning_scene_service,
            callback_group=self.async_group,
        )
        self.place_client = self.create_client(
            PlaceCommand,
            self.names.conveyor_place_service,
            callback_group=self.async_group,
        )

        self.get_logger().info(
            f"supervisor 시작 (execute={self.execute})\n" + self.names.describe()
        )

    # ══════════════════════════════════════════════════════════
    # 세대 관리
    # ══════════════════════════════════════════════════════════
    def on_simulation_state(self, message):
        previous = self.simulation_state
        self.simulation_state = message

        if previous is None or message.reset_id != previous.reset_id:
            self.get_logger().info(
                f"reset_id {message.reset_id}. 대기열과 완료·실패 기록을 폐기합니다."
            )
            self.clear_all()
            # 안전 정지는 scene version 변경으로 풀리지 않고 reset으로만 풀린다.
            self.safety_stopped = False

        self.reset_id = message.reset_id
        self.scene_version = message.scene_version

        if message.state in (SimulationState.STOPPED, SimulationState.INITIALIZING):
            self.cancel_active("308:SIMULATION_RESET")

    def on_scene(self, message):
        self.scene = message
        if message.reset_id != self.reset_id or message.scene_version != self.scene_version:
            self.get_logger().debug(
                "scene 세대가 simulation state와 다릅니다. 최신 snapshot을 요청합니다."
            )
            self.request_snapshot()

    def request_snapshot(self):
        if not self.scene_client.service_is_ready():
            return
        future = self.scene_client.call_async(GetPlanningScene.Request())
        future.add_done_callback(self.on_snapshot_response)

    def on_snapshot_response(self, future):
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"planning scene snapshot 요청 실패: {error}")
            return
        if response is not None and response.success:
            self.scene = response.scene

    def on_perception_status(self, message):
        if message.status != HarvestPerceptionStatus.OK:
            self.get_logger().debug(
                f"perception status={message.status}: {message.message}"
            )

    def on_motion_status(self, message):
        if not message.success and message.error_code:
            self.get_logger().debug(
                f"motion status {message.current_state}: {message.error_code}"
            )

    def clear_all(self):
        self.waiting_for_place_lock = False
        self.pending.clear()
        self.retry_queue.clear()
        self.completed.clear()
        self.failed.clear()
        self.active = None
        self.sequence_index = 0
        self.goal_handle = None

    # ══════════════════════════════════════════════════════════
    # target 검증과 대기열
    # ══════════════════════════════════════════════════════════
    def validate_target(self, message):
        """계획에 쓸 수 있는 target인지 확인한다. 실패 사유 문자열을 돌려준다."""
        if self.simulation_state is None:
            return "SimulationState를 아직 받지 못했습니다."
        if self.simulation_state.state not in (
            SimulationState.READY,
            SimulationState.PLAYING,
        ):
            return f"simulation state={self.simulation_state.state}"
        if message.header.frame_id != "world":
            return f"frame_id가 world가 아닙니다: {message.header.frame_id}"
        if message.reset_id != self.reset_id:
            return f"reset_id 불일치: {message.reset_id} != {self.reset_id}"
        if message.scene_version != self.scene_version:
            return (
                f"scene_version 불일치: {message.scene_version} != {self.scene_version}"
            )
        if not message.target_id:
            return "target_id가 비어 있습니다."

        center = np.array(
            [message.position.x, message.position.y, message.position.z], dtype=float
        )
        if not np.all(np.isfinite(center)):
            return "position이 유효하지 않습니다."

        max_age = optional_threshold(TARGET_MAX_AGE_SEC)
        if max_age is not None:
            age = (self.get_clock().now().nanoseconds - stamp_to_ns(message.header.stamp)) / 1e9
            if age > max_age:
                return f"target이 {age:.2f}초 지났습니다."

        min_confidence = optional_threshold(MIN_TARGET_CONFIDENCE)
        if min_confidence is not None and message.confidence < min_confidence:
            return f"confidence {message.confidence:.2f} 미달"

        min_depth_ratio = optional_threshold(MIN_VALID_DEPTH_RATIO)
        if min_depth_ratio is not None and message.valid_depth_ratio < min_depth_ratio:
            return f"valid_depth_ratio {message.valid_depth_ratio:.2f} 미달"

        max_tf_error = optional_threshold(MAX_TF_TIME_ERROR_SEC)
        if max_tf_error is not None and message.tf_time_error_sec > max_tf_error:
            return f"TF 시간 오차 {message.tf_time_error_sec:.3f}초 초과"

        return ""

    def on_target(self, message):
        if self.safety_stopped:
            return

        reason = self.validate_target(message)
        if reason:
            self.get_logger().debug(f"target {message.target_id} 거부: {reason}")
            return

        key = (message.reset_id, message.target_id)
        if key in self.completed or key in self.failed:
            # 같은 reset_id에서 성공했거나 재시도까지 실패한 target은 다시
            # 실행하지 않는다.
            return

        center = np.array(
            [message.position.x, message.position.y, message.position.z], dtype=float
        )
        stamp_ns = stamp_to_ns(message.header.stamp)

        if self.active is not None and self.active.key == key:
            # 실행 중인 target의 갱신은 실행 목표를 바꾸지 않는다.
            return

        existing = self.pending.get(key)
        if existing is not None and existing.stamp_ns >= stamp_ns:
            return

        self.pending[key] = PendingTarget(
            key=key,
            message=message,
            center=center,
            stamp_ns=stamp_ns,
            apple_id=existing.apple_id if existing else "",
            attempts=existing.attempts if existing else 0,
        )

        # 한 프레임의 target이 다 도착할 때까지 잠깐 모은다.
        if self.batch_timer is None:
            self.batch_timer = self.create_timer(
                TARGET_BATCH_DEBOUNCE_SEC,
                self.on_batch_ready,
                callback_group=self.async_group,
            )

    def on_batch_ready(self):
        if self.batch_timer is not None:
            self.batch_timer.cancel()
            self.batch_timer = None
        self.assign_apple_ids()
        self.start_next_target()

    def assign_apple_ids(self):
        """대기 중 target에 전역 apple ID를 부여한다."""
        unassigned = [item for item in self.pending.values() if not item.apple_id]
        if not unassigned:
            return
        robot_position = self.robot_base_position()
        if robot_position is None:
            return
        try:
            mapping = assign_apple_ids_by_distance(
                self.robot_id, robot_position, list(self.pending.values())
            )
        except ValueError as error:
            self.get_logger().warning(f"apple ID 부여 실패: {error}")
            return
        for key, apple_id in mapping.items():
            if key in self.pending:
                self.pending[key].apple_id = apple_id

    def robot_base_position(self):
        """이 로봇의 world base 위치.

        ``/planning_scene`` 의 ``robot_base_pose`` 는 세계 하나를 기술하는
        전역 토픽이라 로봇이 둘이면 어느 쪽 base 인지 정해지지 않는다.
        그래서 로봇별로 유일한 USD 루트 prim frame 을 먼저 조회한다.
        """
        frame = BASE_FRAME_BY_ROBOT.get(self.robot_id)
        if frame:
            try:
                transform = self.tf_buffer.lookup_transform("world", frame, Time())
            except TransformException:
                transform = None
            if transform is not None:
                translation = transform.transform.translation
                return np.array(
                    [translation.x, translation.y, translation.z], dtype=float
                )
        if self.scene is not None:
            position = self.scene.robot_base_pose.pose.position
            candidate = np.array([position.x, position.y, position.z], dtype=float)
            if np.any(np.abs(candidate) > 1e-9):
                return candidate
        return None

    def current_tcp_pose(self):
        """world 기준 물리 수확 TCP pose.

        palm frame에서 palm 로컬 +Y로 0.0908 m 떨어진 점이다.
        """
        try:
            transform = self.tf_buffer.lookup_transform("world", TCP_FRAME, Time())
        except TransformException:
            return None, None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x, y, z, w = (
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        matrix = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        palm = np.array(
            [translation.x, translation.y, translation.z], dtype=float
        )
        tcp = palm + matrix[:, 1] * PALM_TO_TCP_Y_M
        return tcp, np.array([x, y, z, w], dtype=float)

    # ══════════════════════════════════════════════════════════
    # 실행
    # ══════════════════════════════════════════════════════════
    def select_next(self):
        """robot base에서 가까운 순서로 다음 target을 고른다."""
        if not self.pending:
            if not self.retry_queue:
                return None
            # 일반 대기열이 비면 재시도 대기열을 1회만 처리한다.
            while self.retry_queue:
                candidate = self.retry_queue.popleft()
                if candidate.key in self.completed or candidate.key in self.failed:
                    continue
                return candidate
            return None

        robot_position = self.robot_base_position()
        candidates = list(self.pending.values())
        if robot_position is None:
            return candidates[0]
        return min(
            candidates,
            key=lambda item: float(np.linalg.norm(item.center - robot_position)),
        )

    def start_next_target(self):
        if self.safety_stopped or self.active is not None:
            return
        if not self.execute:
            if self.pending:
                names = ", ".join(
                    f"{item.apple_id or item.key[1]}" for item in self.pending.values()
                )
                self.get_logger().info(f"[관측 전용] 실행 가능한 target: {names}")
                self.pending.clear()
            return

        candidate = self.select_next()
        if candidate is None:
            return
        self.pending.pop(candidate.key, None)

        if not self.action_client.server_is_ready():
            self.action_client.wait_for_server(timeout_sec=1.0)
            if not self.action_client.server_is_ready():
                self.get_logger().warning("Action 서버가 준비되지 않았습니다.")
                self.pending[candidate.key] = candidate
                return

        self.active = candidate
        self.sequence_index = 0
        self.generation += 1
        self.get_logger().info(
            f"수확 시작: {candidate.apple_id or candidate.key[1]} "
            f"center={np.round(candidate.center, 3)}"
        )
        self.send_current_step()

    def build_goal(self, motion_type):
        """단계별 Goal을 만든다.

        GRASP와 RELEASE의 target_pose는 Goal 전송 시점의 현재 pose로 채운다.
        그 두 단계는 이동 없이 그리퍼만 여닫기 때문이다.
        """
        goal = RobotMotion.Goal()
        goal.motion_type = motion_type
        goal.reset_id = self.reset_id
        goal.scene_version = self.scene_version

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"

        if motion_type in (RobotMotion.Goal.GRASP, RobotMotion.Goal.RELEASE):
            tcp, orientation = self.current_tcp_pose()
            if tcp is None:
                # palm frame 을 못 읽어도 이 두 단계는 이동이 없다. GPU PC 1이
                # 자기 articulation 의 실제 현재 pose 를 그대로 유지하므로,
                # 여기 채우는 좌표가 실행 목표를 바꾸지 않는다. 참고용으로
                # 현재 target 중심을 넣는다.
                tcp = self.active.center if self.active is not None else np.zeros(3)
                orientation = np.array([0.0, 0.0, 0.0, 1.0])
            pose.pose.position.x = float(tcp[0])
            pose.pose.position.y = float(tcp[1])
            pose.pose.position.z = float(tcp[2])
            pose.pose.orientation.x = float(orientation[0])
            pose.pose.orientation.y = float(orientation[1])
            pose.pose.orientation.z = float(orientation[2])
            pose.pose.orientation.w = float(orientation[3])
        else:
            center = self.active.center
            pose.pose.position.x = float(center[0])
            pose.pose.position.y = float(center[1])
            pose.pose.position.z = float(center[2])
            robot_position = self.robot_base_position()
            if robot_position is None:
                orientation = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                orientation = approach_orientation_xyzw(robot_position, center)
            pose.pose.orientation.x = float(orientation[0])
            pose.pose.orientation.y = float(orientation[1])
            pose.pose.orientation.z = float(orientation[2])
            pose.pose.orientation.w = float(orientation[3])

        goal.target_pose = pose
        # waypoints는 GPU PC 1 내부 planner가 만든다. 외부에서 주입하지 않는다.
        goal.waypoints = []
        return goal

    def send_current_step(self):
        if self.active is None or self.sequence_index >= len(SEQUENCE):
            return
        motion_type = SEQUENCE[self.sequence_index]

        # 컨베이어는 한 대뿐이다. 배치 구간에 들어가기 전에 락을 잡는다.
        if motion_type == PLACE_LOCK_START and not self.place_locked:
            self.reserve_place_slot()
            return

        try:
            goal = self.build_goal(motion_type)
        except RoutePlanningError as error:
            self.handle_step_failure(motion_type, "310:TF_UNAVAILABLE", str(error))
            return

        if self.scene is not None:
            try:
                validate_scene_version(
                    self.scene.reset_id,
                    self.scene.scene_version,
                    self.reset_id,
                    self.scene_version,
                )
            except RoutePlanningError as error:
                self.handle_step_failure(
                    motion_type, "312:INTERNAL_ERROR", f"scene 세대 불일치: {error}"
                )
                return

        self.get_logger().info(f"  -> {MOTION_NAMES[motion_type]}")
        generation = self.generation
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self.on_feedback
        )
        future.add_done_callback(
            lambda result: self.on_goal_response(result, generation)
        )

    def on_goal_response(self, future, generation):
        if generation != self.generation:
            return
        try:
            handle = future.result()
        except Exception as error:  # noqa: BLE001
            self.handle_step_failure(
                SEQUENCE[self.sequence_index], "312:INTERNAL_ERROR", str(error)
            )
            return
        if handle is None or not handle.accepted:
            self.handle_step_failure(
                SEQUENCE[self.sequence_index],
                "306:GOAL_REJECTED",
                "Action 서버가 Goal을 거부했습니다.",
            )
            return
        self.goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda result: self.on_result(result, generation)
        )

    def on_feedback(self, message):
        feedback = message.feedback
        self.get_logger().debug(
            f"     {feedback.current_state} {feedback.progress * 100:.0f}%"
        )

    def on_result(self, future, generation):
        if generation != self.generation:
            return
        motion_type = SEQUENCE[self.sequence_index]
        try:
            result = future.result().result
        except Exception as error:  # noqa: BLE001
            self.handle_step_failure(motion_type, "312:INTERNAL_ERROR", str(error))
            return

        self.goal_handle = None
        if not result.success:
            self.handle_step_failure(motion_type, result.error_code, result.message)
            return

        self.get_logger().info(f"     {MOTION_NAMES[motion_type]} 완료")

        # RELEASE가 끝나면 컨베이어 락을 놓는다. 다른 로봇이 기다린다.
        if motion_type == PLACE_LOCK_END and self.place_locked:
            self.release_place_slot()

        self.sequence_index += 1
        if self.sequence_index >= len(SEQUENCE):
            self.finish_active(success=True)
            return
        self.send_current_step()

    def handle_step_failure(self, motion_type, error_code, message):
        name = MOTION_NAMES.get(motion_type, "UNKNOWN")
        self.get_logger().error(f"     {name} 실패 [{error_code}] {message}")

        if self.place_locked:
            self.fail_place_slot(error_code, message)

        if motion_type in POST_CONTACT_MOTIONS:
            # 사과를 들었거나 로봇이 나무 안에 있을 수 있다. 다음 Goal을
            # 보내지 않고 안전 정지한다. reset으로만 해제된다.
            self.get_logger().error(
                "접촉 이후 실패입니다. SAFETY_STOPPED로 전환하고 대기열을 폐기합니다."
            )
            self.safety_stopped = True
            if self.active is not None:
                self.failed.add(self.active.key)
            self.pending.clear()
            self.retry_queue.clear()
            self.active = None
            return

        # 접촉 전 첫 실패는 재시도 대기열로 보낸다. 다른 사과를 모두
        # 처리한 뒤 1회만 재시도한다.
        candidate = self.active
        self.active = None
        if candidate is None:
            return
        candidate.attempts += 1
        if candidate.attempts >= 2:
            self.failed.add(candidate.key)
            self.get_logger().error(
                f"{candidate.apple_id or candidate.key[1]} 최종 실패"
            )
        else:
            self.retry_queue.append(candidate)
            self.get_logger().warning(
                f"{candidate.apple_id or candidate.key[1]} 재시도 대기열로 이동"
            )
        self.start_next_target()

    def finish_active(self, success):
        candidate = self.active
        self.active = None
        self.sequence_index = 0
        if candidate is None:
            return
        if success:
            self.completed.add(candidate.key)
            self.get_logger().info(
                f"수확 완료: {candidate.apple_id or candidate.key[1]}"
            )
        self.start_next_target()

    def cancel_active(self, error_code):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        if self.place_locked:
            self.fail_place_slot(error_code, "simulation reset")
        self.active = None
        self.sequence_index = 0

    # ══════════════════════════════════════════════════════════
    # 공유 컨베이어 배치 락
    # ══════════════════════════════════════════════════════════
    def on_place_status(self, message):
        self.place_state = message
        mine = (
            message.lock_owner_robot_id == self.robot_id
            and message.reservation_id == self.reservation_id
        )
        if mine:
            self.place_locked = message.state in (
                PlaceCoordinatorStatus.RESERVED,
                PlaceCoordinatorStatus.PLACING,
                PlaceCoordinatorStatus.LANDING_CHECK,
            )
        # 대기열에 넣어 둔 예약이 내 차례가 되었는지는 이 토픽으로만 알 수
        # 있다. 락 서비스는 승격 시점에 별도 응답을 주지 않는다.
        if (
            self.waiting_for_place_lock
            and mine
            and message.state == PlaceCoordinatorStatus.RESERVED
        ):
            self.begin_placing()

    def place_request(self, command, error_code="", message=""):
        request = PlaceCommand.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = "world"
        request.command = command
        request.robot_id = self.robot_id
        request.reservation_id = self.reservation_id
        request.apple_id = self.active.apple_id if self.active else ""
        request.place_position_id = self.place_position_id
        request.reset_id = self.reset_id
        request.scene_version = self.scene_version
        request.safety_confirmed = True
        request.error_code = error_code
        request.message = message
        return request

    def send_place_command(self, request, callback):
        if not self.place_client.service_is_ready():
            self.place_client.wait_for_service(timeout_sec=PLACE_SERVICE_WAIT_SEC)
        if not self.place_client.service_is_ready():
            self.handle_step_failure(
                PLACE_LOCK_START,
                "312:INTERNAL_ERROR",
                f"{self.names.conveyor_place_service} 서비스를 찾지 못했습니다.",
            )
            return
        future = self.place_client.call_async(request)
        future.add_done_callback(callback)

    def reserve_place_slot(self):
        """컨베이어 배치 슬롯을 예약한다. 락을 받으면 TRANSPORT를 보낸다."""
        self.reservation_id = uuid.uuid4().hex[:12]
        self.waiting_for_place_lock = True
        self.get_logger().info(
            f"  -> 컨베이어 배치 예약 요청 ({self.place_position_id})"
        )
        self.send_place_command(
            self.place_request(PlaceCommand.Request.RESERVE), self.on_reserve_response
        )

    def begin_placing(self):
        """락을 실제로 소유한 뒤에만 배치 구간을 시작한다."""
        if not self.waiting_for_place_lock:
            return
        self.waiting_for_place_lock = False
        self.place_locked = True
        self.get_logger().info("     컨베이어 배치 슬롯 확보")
        self.send_place_command(
            self.place_request(PlaceCommand.Request.START_PLACING),
            self.on_start_placing_response,
        )

    def on_reserve_response(self, future):
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.handle_step_failure(
                PLACE_LOCK_START, "312:INTERNAL_ERROR", str(error)
            )
            return
        if response is None or not response.accepted:
            self.waiting_for_place_lock = False
            self.handle_step_failure(
                PLACE_LOCK_START,
                response.error_code if response else "312:INTERNAL_ERROR",
                response.message if response else "예약 응답이 없습니다.",
            )
            return
        if response.queued:
            # 대기열 등록도 accepted=True 로 온다. queued 를 먼저 보지 않으면
            # 락을 받지 못한 채 START_PLACING 을 보내서 LOCK_OWNER_MISMATCH
            # 또는 INVALID_STATE 로 실패한다. 두 로봇이 거의 동시에 PULL 을
            # 끝냈을 때 실제로 그렇게 터졌다.
            self.get_logger().info(
                "     컨베이어 사용 중. 대기열에 등록했고 차례를 기다립니다."
            )
            return
        self.begin_placing()

    def on_start_placing_response(self, future):
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.handle_step_failure(
                PLACE_LOCK_START, "312:INTERNAL_ERROR", str(error)
            )
            return
        if response is None or not response.accepted:
            self.handle_step_failure(
                PLACE_LOCK_START,
                response.error_code if response else "312:INTERNAL_ERROR",
                response.message if response else "배치 시작 응답이 없습니다.",
            )
            return
        # 락을 잡았으니 TRANSPORT부터 이어서 실행한다.
        self.send_current_step_after_lock()

    def send_current_step_after_lock(self):
        """락 확보 뒤 TRANSPORT Goal을 실제로 보낸다."""
        if self.active is None:
            return
        motion_type = SEQUENCE[self.sequence_index]
        try:
            goal = self.build_goal(motion_type)
        except RoutePlanningError as error:
            self.handle_step_failure(motion_type, "310:TF_UNAVAILABLE", str(error))
            return
        self.get_logger().info(f"  -> {MOTION_NAMES[motion_type]}")
        generation = self.generation
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self.on_feedback
        )
        future.add_done_callback(
            lambda result: self.on_goal_response(result, generation)
        )

    def release_place_slot(self):
        """배치 구간을 끝내고 락을 완전히 놓는다.

        락 서비스의 정상 흐름은 RESERVE -> START_PLACING -> RELEASED ->
        CONFIRM_LANDING 이다. RELEASED 는 상태를 LANDING_CHECK 로 옮길 뿐이고,
        락을 IDLE 로 되돌려 다음 로봇을 승격시키는 것은 CONFIRM_LANDING 이다.
        여기서 멈추면 robot_01 이 컨베이어를 계속 점유해서 robot_02 는
        수확을 끝내고도 배치 예약을 못 받는다.

        reservation_id 는 CONFIRM_LANDING 이 끝날 때까지 유지해야 한다.
        락 서비스가 소유자 확인에 그 값을 쓴다.
        """
        self.get_logger().info("     컨베이어 배치 완료 보고 (RELEASED)")
        self.send_place_command(
            self.place_request(PlaceCommand.Request.RELEASED),
            self.on_released_response,
        )

    def on_released_response(self, future):
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"배치 반납 응답 실패: {error}")
            self.place_locked = False
            self.waiting_for_place_lock = False
            return
        if response is None or not response.accepted:
            self.get_logger().warning(
                f"RELEASED 거절: "
                f"{response.error_code if response else ''} "
                f"{response.message if response else ''}"
            )
            self.place_locked = False
            self.waiting_for_place_lock = False
            return
        self.get_logger().info("     착지 확인 보고 (CONFIRM_LANDING)")
        self.send_place_command(
            self.place_request(PlaceCommand.Request.CONFIRM_LANDING),
            self.on_confirm_landing_response,
        )

    def on_confirm_landing_response(self, future):
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"착지 확인 응답 실패: {error}")
        else:
            if response is not None and response.accepted:
                self.get_logger().info("     컨베이어 락 반납 완료")
            else:
                self.get_logger().warning(
                    f"CONFIRM_LANDING 거절: "
                    f"{response.error_code if response else ''}"
                )
        self.place_locked = False
        self.waiting_for_place_lock = False
        self.reservation_id = ""

    def fail_place_slot(self, error_code, message):
        """실패를 보고하고 락을 풀어 준다.

        FAIL 은 상태를 ERROR 로 옮기기만 한다. ERROR 에서 락을 놓으려면
        CLEAR_ERROR 를 안전 확인과 함께 보내야 한다. 이게 빠지면 실패한
        로봇이 컨베이어를 영구히 잡고 있게 된다.
        """
        self.send_place_command(
            self.place_request(PlaceCommand.Request.FAIL, error_code, message),
            self.on_fail_place_response,
        )

    def on_fail_place_response(self, future):
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"배치 실패 보고 응답 실패: {error}")
            self.place_locked = False
            self.waiting_for_place_lock = False
            self.reservation_id = ""
            return
        self.send_place_command(
            self.place_request(PlaceCommand.Request.CLEAR_ERROR),
            self.on_clear_error_response,
        )

    def on_clear_error_response(self, future):
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"배치 오류 해제 응답 실패: {error}")
        self.place_locked = False
        self.waiting_for_place_lock = False
        self.reservation_id = ""


def main():
    parser = argparse.ArgumentParser(description="GPU PC 1 수확 supervisor")
    add_robot_id_argument(parser)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="RobotMotion Goal을 실제로 보낸다. 없으면 관측만 한다.",
    )
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0], *remaining])
    node = None
    executor = None
    try:
        node = HarvestCoordinator(parsed.robot_id, parsed.execute)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
