"""M0617 + Robotiq 3F 그리퍼 사과 수확 코어 (GPU PC 1).

이 파일은 두 가지 역할을 한다.

1. 단독 실행 시 한 대의 M0617으로 사과 하나를 수확해 컨베이어에 놓는다.
2. ``vision_apple_pick.py``가 import 하는 물리·IK·계획 코어 모듈이다.

씬에는 M0617이 두 대 있고 컨베이어는 하나다. 둘 다 같은 세계에서 움직여야
하므로, 로봇에 딸린 상태는 모듈 전역이 아니라 :class:`RobotRuntime` 하나에
모은다. 한 프로세스가 런타임을 두 개 만들면 한 Isaac Sim 안에서 두 로봇이
동시에 일한다. 모듈 전역 경로는 단독 실행 편의를 위한 기본값일 뿐이다.

실행:
    /home/rokey/isaacsim/python.sh apple_pick.py --robot-id robot_01

헤드리스 점검:
    /home/rokey/isaacsim/python.sh apple_pick.py --headless --max-steps 600

동작 순서 (docs/features/harvesting.md 상태 흐름):
    TARGET_RECEIVED -> PRE_GRASP_PLANNING -> APPROACH -> GRASP -> TWIST
    -> LINEAR_PULL -> STEM_BREAK_CHECK -> TRANSPORT -> PLACE -> RELEASE
    -> RETRACT

중요:
    * 이 코드는 저장된 USD를 수정하거나 저장하지 않는다. 실행 중 변경은
      전부 Session Layer에만 적용한다.
    * FixedJoint 파손 한계(15 N / 2 N·m)는 당기는 명령값이 아니라 joint가
      끊어지는 반력 한계다.
    * 구동하지 않는 로봇은 저장된 자세를 그대로 유지하도록 고정한다.
"""

import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


# ══════════════════════════════════════════════════════════════
# SimulationApp은 다른 Isaac Sim 모듈보다 먼저 만들어야 한다.
# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="M0617 3F apple picking")
parser.add_argument("--headless", action="store_true", help="화면 없이 실행")
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="0이면 창을 닫을 때까지, 양수이면 지정한 물리 스텝 뒤 종료",
)
parser.add_argument(
    "--robot-id",
    choices=("robot_01", "robot_02"),
    default="robot_01",
    help="단독 실행에서 수확할 USD 로봇 프로파일 (기본값: robot_01)",
)
parser.add_argument(
    "--break-torque-nm",
    type=float,
    default=2.0,
    help="stem FixedJoint break torque (docs 기준 2.0 N·m)",
)
parser.add_argument(
    "--entry-clearance-m",
    type=float,
    default=None,
    help=(
        "entry pre-shape 사전검사에서 요구할 최소 swept clearance. "
        "진입 중 실시간 중단 기준보다 작게 지정할 수 없다."
    ),
)
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "sync_loads": True,
        "width": 1280,
        "height": 720,
    }
)

import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.bindings._physx import SimulationEvent  # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualSphere  # noqa: E402
from isaacsim.core.utils.rotations import (  # noqa: E402
    quat_to_rot_matrix,
    rot_matrix_to_quat,
)
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.robot.manipulators.manipulators import SingleManipulator  # noqa: E402
from isaacsim.robot_motion.motion_generation import (  # noqa: E402
    ArticulationMotionPolicy,
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    RmpFlow,
)
from isaacsim.robot_motion.motion_generation.lula import RRT  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 파일 경로
# ══════════════════════════════════════════════════════════════
PROJECT_DIR = Path(__file__).resolve().parent
STAGE_PATH = PROJECT_DIR / "m0617_3fgripper08201638.usd"
DESCRIPTION_PATH = PROJECT_DIR / "m0617_robot_description.yaml"
RMPFLOW_CONFIG_PATH = PROJECT_DIR / "m0617_rmpflow_config.yaml"
RRT_CONFIG_PATH = PROJECT_DIR / "m0617_rrt_config.yaml"
URDF_PATH = (
    PROJECT_DIR / "m0617_gripper" / "dsr_description2" / "urdf" / "m0617.urdf"
)


# ══════════════════════════════════════════════════════════════
# USD 로봇 프로파일
#
# 각 로봇의 Articulation root는 본체가 아니라 레일의 root_joint이고, 본체는
# 자기 FixedJoint로 레일 마운트에 붙는다. 경로는
# docs/architecture/system_overview.md의 매핑표와 같다.
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RobotRuntimeProfile:
    """저장된 USD에서 한 로봇이 소유하는 실행 자산 경로 모음."""

    robot_id: str
    xform_root_path: str
    articulation_prim_path: str
    robot_prim_path: str
    apple_parent_path: str
    tree_root_path: str
    camera_root_path: str
    initial_arm_joints_deg: tuple

    @property
    def articulation_root_joint_path(self):
        return f"{self.articulation_prim_path}/root_joint"

    @property
    def rail_joint_path(self):
        return f"{self.articulation_prim_path}/joints/rail_joint"

    @property
    def robot_mount_joint_path(self):
        return f"{self.robot_prim_path}/FixedJoint"

    @property
    def base_path(self):
        return f"{self.robot_prim_path}/base_link"

    @property
    def link6_path(self):
        return f"{self.robot_prim_path}/link_6"

    @property
    def gripper_root_path(self):
        return f"{self.robot_prim_path}/robotiq_3f_gripper_articulated"

    @property
    def palm_path(self):
        return f"{self.gripper_root_path}/palm"

    @property
    def camera_prim_path(self):
        # 껍데기가 아니라 실제로 렌더링하는 자식 prim이다. 껍데기를 돌려도
        # 광학 프레임은 payload 안의 자체 변환 때문에 예측대로 향하지 않는다.
        return f"{self.camera_root_path}/RSD455/Camera_OmniVision_OV9782_Color"

    @property
    def apple_assembly_root_paths(self):
        return tuple(
            f"{self.apple_parent_path}/apple_branch{suffix}"
            for suffix in ("", "_1", "_2")
        )

    @property
    def initial_arm_joints_rad(self):
        return np.deg2rad(np.array(self.initial_arm_joints_deg, dtype=float))


ROBOT_RUNTIME_PROFILES = {
    "robot_01": RobotRuntimeProfile(
        robot_id="robot_01",
        xform_root_path="/World/Xform_01",
        articulation_prim_path="/World/Xform_01/m0617_rail",
        robot_prim_path="/World/Xform_01/m0617_01",
        apple_parent_path="/World/Xform",
        tree_root_path="/World/Xform/tree",
        camera_root_path="/World/base_rsd455_01",
        initial_arm_joints_deg=(0.0, 0.0, -90.0, 0.0, 90.0, 0.0),
    ),
    "robot_02": RobotRuntimeProfile(
        robot_id="robot_02",
        xform_root_path="/World/Xform_02",
        articulation_prim_path="/World/Xform_02/m0617_rail",
        robot_prim_path="/World/Xform_02/m0617_02",
        apple_parent_path="/World/Xform_03",
        tree_root_path="/World/Xform_03/tree",
        camera_root_path="/World/base_rsd455_02",
        initial_arm_joints_deg=(0.0, 0.0, 90.0, 0.0, -90.0, 0.0),
    ),
}
ROBOT_IDS = tuple(ROBOT_RUNTIME_PROFILES)

# 단독 실행 편의를 위한 기본 프로파일이다. 한 프로세스에서 두 대를 굴릴
# 때는 이 전역을 쓰지 않고 RobotRuntime 을 로봇마다 하나씩 만든다.
ROBOT_PROFILE = ROBOT_RUNTIME_PROFILES[args.robot_id]
ARTICULATION_PRIM_PATH = ROBOT_PROFILE.articulation_prim_path
ARTICULATION_ROOT_JOINT_PATH = ROBOT_PROFILE.articulation_root_joint_path
ROBOT_PRIM_PATH = ROBOT_PROFILE.robot_prim_path
ROBOT_BASE_PATH = ROBOT_PROFILE.base_path
LINK6_PATH = ROBOT_PROFILE.link6_path
GRIPPER_ROOT_PATH = ROBOT_PROFILE.gripper_root_path
PALM_PATH = ROBOT_PROFILE.palm_path
TREE_ROOT_PATH = ROBOT_PROFILE.tree_root_path
INITIAL_ARM_JOINTS_RAD = ROBOT_PROFILE.initial_arm_joints_rad

# 런타임에 만드는 prim. 저장된 stage에는 없는 것이 정상이다.
PLANNING_OBSTACLE_ROOT_PATH = "/World/RuntimeHarvestPlanningObstacles"
RUNTIME_TREE_COLLIDER_ROOT_PATH = "/World/RuntimeHarvestTreeColliders"
CONVEYOR_PATH = "/World/ConveyorTrack_01"
RUNTIME_CONVEYOR_COLLIDER_PATH = "/World/RuntimeConveyorBeltSurface"

SHOW_PLANNING_DEBUG = os.environ.get("HARVEST_SHOW_PLANNING_DEBUG", "1") != "0"


# ══════════════════════════════════════════════════════════════
# 관절
# ══════════════════════════════════════════════════════════════
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# 이 3F 모델에는 mimic joint가 없으므로 11개 회전관절을 직접 동기화한다.
# palm_finger_middle_joint는 USD에서 Fixed라 구동 대상이 아니다.
GRIPPER_JOINTS = [
    "palm_finger_1_joint",
    "finger_1_joint_1",
    "finger_1_joint_2",
    "finger_1_joint_3",
    "palm_finger_2_joint",
    "finger_2_joint_1",
    "finger_2_joint_2",
    "finger_2_joint_3",
    "finger_middle_joint_1",
    "finger_middle_joint_2",
    "finger_middle_joint_3",
]


# ══════════════════════════════════════════════════════════════
# 물리 상수 (docs/features/harvesting.md, asset_requirements.md)
# ══════════════════════════════════════════════════════════════
APPLE_DIAMETER_M = 0.080
APPLE_RADIUS_M = 0.040

# 물리 수확 TCP: palm 원점에서 palm 로컬 +Y 0.0908 m.
# palm collision mesh 앞면 0.0508 m + 명목 사과 반지름 0.040 m.
PALM_TO_TCP_Y_M = 0.0908

STEM_BREAK_FORCE_N = 15.0
STEM_BREAK_TORQUE_NM = float(args.break_torque_nm)

PREGRASP_DISTANCE_M = 0.15
STAGING_DISTANCE_M = 0.30
FINAL_APPROACH_DISTANCE_M = 0.03

TWIST_ANGLE_DEG = 45.0
TWIST_DURATION_S = 1.0
PULL_SPEED_MPS = 0.050
PULL_MAX_DISTANCE_M = 0.100

# 컨베이어 1 상면 위 30 mm 이하로 낮춘 뒤 놓는다.
PLACE_HEIGHT_ABOVE_BELT_M = 0.030

# 유의미한 진전이 이 시간 동안 없으면 timeout. Pause 중에는 sim time이
# 흐르지 않으므로 watchdog도 함께 멈춘다.
MOTION_TIMEOUT_S = 3.0

TARGET_POSITION_TOLERANCE_M = 0.005
TARGET_ORIENTATION_TOLERANCE_DEG = 3.0

GRASP_SETTLE_STEPS = 120
TWIST_STEPS = 60

RMPFLOW_MAXIMUM_SUBSTEP_S = 1.0 / 300.0
RMPFLOW_SEGMENT_STEPS = 360
# 나무에서 컨베이어까지는 이동 거리가 길어 360 step(6초)으로는 도착 전에
# 한도에 걸린다. 정체는 RMPFLOW_STALL_STEPS 감시가 따로 잡으므로, 예산을
# 늘려도 진짜로 멈춘 경우는 여전히 빨리 실패한다.
TRANSIT_SEGMENT_STEPS = RMPFLOW_SEGMENT_STEPS * 3
RMPFLOW_STALL_STEPS = 120
RMPFLOW_STALL_POSITION_DELTA_M = 0.005
RMPFLOW_STALL_ROTATION_DELTA_DEG = 2.0

# Lula trajectory 시험용 임시 제한. 정식값 승인 전까지 TBD.
RRT_TRAJECTORY_VELOCITY_LIMITS = np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.0])
RRT_TRAJECTORY_ACCELERATION_LIMITS = np.array([1.5, 1.5, 2.0, 3.0, 3.0, 3.0])
RRT_TRAJECTORY_JERK_LIMITS = np.array([20.0, 20.0, 25.0, 40.0, 40.0, 40.0])
RRT_TRAJECTORY_SAMPLE_DT_S = 1.0 / 60.0

# planning proxy (docs/assets/asset_requirements.md)
TREE_SEGMENT_LENGTH_M = 0.040
TREE_MIN_PCA_RADIUS_M = 0.020

# 회피 대상은 몸통과 가지뿐이다. 잎·지면·열매는 visual-only 로 두고 PhysX
# collision 과 planning obstacle 양쪽에서 뺀다. 부딪혀도 되는 것을 장애물로
# 넣으면 사과 앞까지 가는 길이 거의 다 막힌다. summerTree 자산에서 구조
# mesh 는 `summerTree` 하나이고 나머지가 전부 여기에 걸린다.
TREE_NON_STRUCTURAL_KEYWORDS = (
    "leaves",
    "leawes",
    "leaf",
    "berries",
    "berry",
    "ground",
    "grass",
)

PROXY_DEFAULT_SPHERE_RADIUS_M = 0.010
ROBOT_TREE_SAFETY_MARGIN_M = 0.050
MAX_BRANCH_PROXIES = 48
TREE_SIGNATURE_QUANTUM_M = 0.02

# Drive. 충돌 시 과도한 강성이 컨베이어를 뚫지 않도록 제한한다.
ARM_DRIVE_STIFFNESS = 1.0e6
ARM_DRIVE_DAMPING = 1.0e4
ARM_DRIVE_MAX_FORCE = 2.0e3
GRIPPER_DRIVE_STIFFNESS = 50.0
GRIPPER_DRIVE_DAMPING = 5.0

# 11개 손가락 관절의 접촉 토크가 stem의 2 N·m 한계에 집중되지 않도록
# GRASP는 저토크로 닿고, TWIST/PULL과 운반 중에는 미끄러지지 않도록 올린다.
GRIPPER_GRASP_MAX_FORCE = float(os.environ.get("HARVEST_GRASP_FORCE", "0.08"))
GRIPPER_HOLD_MAX_FORCE = float(os.environ.get("HARVEST_HOLD_FORCE", "0.50"))
GRIPPER_DRIVE_MAX_FORCE = GRIPPER_GRASP_MAX_FORCE
# entry pre-shape는 사과에 닿기 전 자세라 stem 한계와 무관하다. 저토크로는
# 팔이 가속하는 동안 손가락이 명령 자세를 유지하지 못해, 정적으로 측정한
# swept clearance와 실제 진입 자세가 달라진다.
GRIPPER_ENTRY_MAX_FORCE = GRIPPER_HOLD_MAX_FORCE

# 진입 중 실시간으로 다시 잰 swept clearance가 이 값 아래로 내려가면 멈춘다.
# 검사 간격 동안 진행하는 거리보다 커야 접촉 전에 멈출 수 있다.
ENTRY_LIVE_MIN_CLEARANCE_M = 0.004
ENTRY_SWEEP_MIN_CLEARANCE_M = 0.005
if args.entry_clearance_m is not None:
    _requested = float(args.entry_clearance_m)
    if _requested < ENTRY_LIVE_MIN_CLEARANCE_M:
        raise SystemExit(
            f"--entry-clearance-m 은 진입 중 중단 기준 "
            f"{ENTRY_LIVE_MIN_CLEARANCE_M} m 이상이어야 합니다: {_requested}"
        )
    ENTRY_SWEEP_MIN_CLEARANCE_M = _requested
ENTRY_LIVE_CHECK_INTERVAL_STEPS = 10
ENTRY_PRESHAPE_SAMPLE_STEPS = 8
# 명목 지름 80 mm 사과에서 손가락 안쪽 면이 중심에서 양쪽 50 mm.
ENTRY_TARGET_HALF_OPENING_M = 0.050
APPLE_GRASP_MAX_DISTANCE_M = 0.060


# ══════════════════════════════════════════════════════════════
# 그리퍼 자세
# ══════════════════════════════════════════════════════════════
# URDF(robotiq_3f_isaac.urdf)의 손가락 관절 한계. 진입 여유를 넓히는 방향은
# proximal/medial을 펴고(0) distal을 최대로 접는(음의 한계) 쪽뿐이다.
FINGER_JOINT_1_LOWER_RAD = 0.0
FINGER_JOINT_2_LOWER_RAD = 0.0
FINGER_JOINT_3_LOWER_RAD = -1.2217304764
FINGER_LIMIT_MARGIN_RAD = 0.005
ENTRY_DISTAL_MAX_RAD = FINGER_JOINT_3_LOWER_RAD + FINGER_LIMIT_MARGIN_RAD


def _entry_preshape(spread_rad, distal_rad):
    """두 측면 palm joint와 세 distal의 entry 자세를 만든다."""
    spread_rad = float(spread_rad)
    return np.array(
        [
            spread_rad,
            FINGER_JOINT_1_LOWER_RAD,
            FINGER_JOINT_2_LOWER_RAD,
            distal_rad,
            -spread_rad,
            FINGER_JOINT_1_LOWER_RAD,
            FINGER_JOINT_2_LOWER_RAD,
            distal_rad,
            FINGER_JOINT_1_LOWER_RAD,
            FINGER_JOINT_2_LOWER_RAD,
            distal_rad,
        ],
        dtype=float,
    )


# 초기 reset용 최대 개방 자세. 세 distal은 음의 limit 근처로 접어 전방으로
# 길게 뻗은 손가락 끝을 사과 통로 바깥으로 뺀다. all-zero 자세는 palm보다
# 약 0.11 m 앞에서 먼저 선접촉한다.
GRIPPER_OPEN = _entry_preshape(0.25, -1.20)

ENTRY_PALM_SPREAD_CANDIDATES_RAD = (0.10, 0.15, 0.20, 0.25)
ENTRY_DISTAL_CANDIDATES_RAD = (-1.20, ENTRY_DISTAL_MAX_RAD)
GRIPPER_ENTRY_CANDIDATES = [
    (
        f"spread_{spread:.2f}_distal_{distal:.4f}",
        _entry_preshape(spread, distal),
    )
    for spread in ENTRY_PALM_SPREAD_CANDIDATES_RAD
    for distal in ENTRY_DISTAL_CANDIDATES_RAD
]

# 약 80 mm 사과를 감싸기 위한 목표값. 충돌이 정상이라면 손가락은 사과
# 표면에서 멈추고, Drive가 이 목표를 유지해 파지력을 만든다.
GRIPPER_CLOSED = np.array(
    [0.0, 0.75, 0.90, -0.55, 0.0, 0.75, 0.90, -0.55, 0.75, 0.90, -0.55],
    dtype=float,
)


# ══════════════════════════════════════════════════════════════
# 예외
# ══════════════════════════════════════════════════════════════
class HarvestError(RuntimeError):
    """수확 실행 중 발생한 복구 불가 오류."""

    error_code = "312:INTERNAL_ERROR"


class ApproachUnreachableError(HarvestError):
    error_code = "301:APPROACH_UNREACHABLE"


class IkFailedError(HarvestError):
    error_code = "300:IK_FAILED"


class CollisionRiskError(HarvestError):
    error_code = "302:COLLISION_RISK"


class MotionTimeoutError(HarvestError):
    error_code = "304:MOTION_TIMEOUT"


class StemNotBrokenError(HarvestError):
    error_code = "305:STEM_NOT_BROKEN"


class UnexpectedContactError(HarvestError):
    error_code = "302:COLLISION_RISK"


# ══════════════════════════════════════════════════════════════
# 수학 유틸
# ══════════════════════════════════════════════════════════════
def vec(*values):
    return np.array(values, dtype=float)


def normalized(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("길이가 0인 벡터는 정규화할 수 없습니다.")
    return vector / length


def smoothstep(alpha):
    """0에서 1까지 속도가 0으로 시작하고 끝나는 보간 계수."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def rotation_about_axis(axis, angle_rad):
    """Rodrigues 회전행렬."""
    axis = normalized(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def rotation_error_deg(matrix_a, matrix_b):
    """두 회전행렬 사이의 각도 오차(도)."""
    relative = np.asarray(matrix_a, dtype=float).T @ np.asarray(matrix_b, dtype=float)
    cos_angle = (np.trace(relative) - 1.0) * 0.5
    return float(np.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0)))))


def quat_wxyz_to_xyzw(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array([x, y, z, w], dtype=float)


def quat_xyzw_to_wxyz(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float)
    return np.array([w, x, y, z], dtype=float)


def point_to_line_distance(point, start, end):
    """선분 start-end 와 점 사이의 최단 거리."""
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-18:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def make_approach_rotation_for_direction(direction):
    """접근축을 palm 로컬 +Y 에 정렬하는 회전행렬을 만든다.

    수확 TCP가 palm 로컬 +Y 이므로, 사과를 향하는 방향이 그 축과 같아야
    palm 이 사과 뒷면에 닿는 포위 파지가 된다.
    """
    forward = normalized(direction)
    reference = vec(0.0, 0.0, 1.0)
    if abs(float(np.dot(forward, reference))) > 0.95:
        reference = vec(1.0, 0.0, 0.0)
    right = normalized(np.cross(reference, forward))
    # Z = X x Y 여야 오른손 좌표계다. cross(forward, right) 로 쓰면 부호가
    # 뒤집혀 행렬식이 -1 인 반사 행렬이 나온다. 그런 행렬은 회전이 아니라서
    # 쿼터니언으로 바꾸면 값이 망가지고, IK 는 도달할 수 없는 자세를 목표로
    # 받는다. 실측에서 위치 오차는 0.1 mm 인데 자세 오차만 정확히 90도에서
    # 멈추는 증상으로 나타났다.
    up = np.cross(right, forward)
    # 열이 palm 로컬 X, Y, Z 축의 world 표현이다.
    return np.column_stack((right, forward, up))


# ══════════════════════════════════════════════════════════════
# Stage 접근 유틸
# ══════════════════════════════════════════════════════════════
def open_project_stage():
    """저장된 프로젝트 stage를 열고 로드가 끝날 때까지 기다린다."""
    if not STAGE_PATH.exists():
        raise HarvestError(f"USD stage를 찾을 수 없습니다: {STAGE_PATH}")
    context = omni.usd.get_context()
    context.open_stage(str(STAGE_PATH))
    for _ in range(4000):
        simulation_app.update()
        if not is_stage_loading():
            break
    stage = context.get_stage()
    if stage is None:
        raise HarvestError("Stage를 열지 못했습니다.")
    return stage


def require_prim(stage, path):
    """없으면 즉시 실패한다. 조용히 진행하면 원인을 훨씬 뒤에서 만난다."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise HarvestError(f"Prim을 찾을 수 없습니다: {path}")
    return prim


def get_prim_world_pose(stage, path):
    """(position, rotation_matrix) 를 world 기준으로 돌려준다."""
    prim = require_prim(stage, path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationMatrix()
    return (
        np.array([translation[0], translation[1], translation[2]], dtype=float),
        np.array(
            [[rotation[r][c] for c in range(3)] for r in range(3)], dtype=float
        ).T,
    )


def compute_live_prim_center(stage, path):
    """현재 시점의 prim world bounding box 중심."""
    prim = require_prim(stage, path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        position, _ = get_prim_world_pose(stage, path)
        return position
    center = box.GetMidpoint()
    return np.array([center[0], center[1], center[2]], dtype=float)


def prim_world_points(stage, path, max_points=20000, exclude_keywords=()):
    """prim 하위 mesh 정점을 world 좌표 배열로 모은다.

    ``exclude_keywords`` 에 걸리는 경로의 mesh 는 건너뛴다. 나무에서 잎을
    빼는 데 쓴다. 잎은 정점 수가 구조 mesh 의 두 배를 넘어서, 걸러내지
    않으면 표본 상한이 잎으로 다 차고 몸통과 가지는 한 점도 안 들어온다.
    """
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        return np.zeros((0, 3), dtype=float)
    keywords = tuple(str(k).lower() for k in exclude_keywords)
    collected = []
    total = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        lowered = str(prim.GetPath()).lower()
        if any(keyword in lowered for keyword in keywords):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            continue
        points = points_attr.Get()
        if not points:
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        # 정점이 많으면 균일 표본으로 줄인다. 전량을 쓰면 proxy 생성이
        # 프레임 예산을 넘긴다.
        if local.shape[0] > 4000:
            stride = int(np.ceil(local.shape[0] / 4000.0))
            local = local[::stride]
        homogeneous = np.column_stack([local, np.ones(local.shape[0])])
        transform = np.array(
            [[matrix[r][c] for c in range(4)] for r in range(4)], dtype=float
        )
        world = homogeneous @ transform
        collected.append(world[:, :3])
        total += world.shape[0]
        if total >= max_points:
            break
    if not collected:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(collected)


def tree_scene_signature(stage, tree_root_path):
    """나무가 실제로 옮겨졌는지 판정할 양자화된 서명.

    사과를 딸 때 생기는 가지 흔들림보다는 크고, 나무를 실제로 옮겼을
    때보다는 작은 양자로 반올림한다.
    """
    prim = stage.GetPrimAtPath(tree_root_path)
    if not prim.IsValid():
        return ("missing", tree_root_path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return ("empty", tree_root_path)
    low = box.GetMin()
    high = box.GetMax()
    values = [low[0], low[1], low[2], high[0], high[1], high[2]]
    return tuple(
        int(round(float(value) / TREE_SIGNATURE_QUANTUM_M)) for value in values
    )


def session_edit(stage):
    """저장된 USD가 아니라 현재 실행에만 적용되는 편집 컨텍스트."""
    return Usd.EditContext(stage, stage.GetSessionLayer())


# ══════════════════════════════════════════════════════════════
# 사과 assembly
# ══════════════════════════════════════════════════════════════
def configure_breakable_joint(stage, joint_path, break_force, break_torque):
    """stem FixedJoint에 파손 한계를 Session Layer로 적용한다."""
    prim = require_prim(stage, joint_path)
    joint = UsdPhysics.Joint(prim)
    if not joint:
        raise HarvestError(f"Physics Joint가 아닙니다: {joint_path}")
    with session_edit(stage):
        joint.CreateBreakForceAttr().Set(float(break_force))
        joint.CreateBreakTorqueAttr().Set(float(break_torque))
    return prim


def discover_apple_assemblies(stage, profile):
    """branchbody -> applebody 를 잇는 실제 FixedJoint 경로를 찾는다.

    이름을 가정하지 않고 relationship 을 검사한다. 실제 stage 의 joint 는
    ``FixedJoint``, ``FixedJoint_01``, ``FixedJoint_02`` 처럼 번호가 붙어
    있어서, 경로를 문자열로 만들면 두 개가 조용히 어긋난다.
    """
    assemblies = []
    for root_path in profile.apple_assembly_root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        apple_path = f"{root_path}/applebody/apple1"
        branch_path = f"{root_path}/branchbody"
        if not stage.GetPrimAtPath(apple_path).IsValid():
            continue
        joint_path = None
        for prim in Usd.PrimRange(root):
            if "Joint" not in str(prim.GetTypeName()):
                continue
            joint = UsdPhysics.Joint(prim)
            if not joint:
                continue
            targets = [str(t) for t in joint.GetBody0Rel().GetTargets()]
            targets += [str(t) for t in joint.GetBody1Rel().GetTargets()]
            if any(branch_path in target for target in targets) and any(
                "applebody" in target for target in targets
            ):
                joint_path = str(prim.GetPath())
                break
        assemblies.append(
            {
                "root": root_path,
                "apple": apple_path,
                "applebody": f"{root_path}/applebody",
                "branchbody": branch_path,
                "joint": joint_path,
            }
        )
    return tuple(assemblies)


# ══════════════════════════════════════════════════════════════
# Drive 설정
# ══════════════════════════════════════════════════════════════
def configure_joint_drives(stage, profile):
    """한 로봇의 팔·그리퍼 Drive 이득을 설정한다."""
    root = require_prim(stage, profile.robot_prim_path)
    arm_count = 0
    gripper_count = 0
    with session_edit(stage):
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name in ARM_JOINTS:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr().Set(ARM_DRIVE_STIFFNESS)
                drive.CreateDampingAttr().Set(ARM_DRIVE_DAMPING)
                drive.CreateMaxForceAttr().Set(ARM_DRIVE_MAX_FORCE)
                arm_count += 1
            elif name in GRIPPER_JOINTS:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr().Set(GRIPPER_DRIVE_STIFFNESS)
                drive.CreateDampingAttr().Set(GRIPPER_DRIVE_DAMPING)
                drive.CreateMaxForceAttr().Set(GRIPPER_DRIVE_MAX_FORCE)
                gripper_count += 1
    if arm_count != len(ARM_JOINTS):
        raise HarvestError(
            f"{profile.robot_id}: 팔 관절 Drive {len(ARM_JOINTS)}개를 기대했지만 "
            f"{arm_count}개를 찾았습니다."
        )
    if gripper_count != len(GRIPPER_JOINTS):
        raise HarvestError(
            f"{profile.robot_id}: 그리퍼 관절 Drive {len(GRIPPER_JOINTS)}개를 "
            f"기대했지만 {gripper_count}개를 찾았습니다."
        )
    print(
        f"   [{profile.robot_id}] Drive 설정: 팔 {arm_count}, "
        f"그리퍼 {gripper_count}, grasp max force {GRIPPER_DRIVE_MAX_FORCE:.2f} N·m"
    )


def set_gripper_drive_max_force(stage, profile, max_force):
    """그리퍼 유지 토크를 단계별로 바꾼다."""
    root = require_prim(stage, profile.gripper_root_path)
    with session_edit(stage):
        for prim in Usd.PrimRange(root):
            if prim.GetName() in GRIPPER_JOINTS:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateMaxForceAttr().Set(float(max_force))


def hold_idle_robots(stage, driven_robot_ids):
    """구동하지 않는 로봇을 저장된 자세 그대로 고정한다.

    두 로봇 모두 ArticulationRootAPI 를 가져 PhysX 가 둘 다 시뮬레이션한다.
    방치하면 drive target 이 없어 팔이 무너지고 레일이 미끄러진다.

    주의: 방치 로봇에 프로파일의 초기 자세를 쓰면 안 된다. 저장된 현재
    자세와 다르면 Play 순간 그 자세로 순간이동한다. state 를 읽어 그 값을
    target 으로 삼고 state 자체는 건드리지 않는다.

    두 로봇을 모두 구동하면 이 함수는 아무 일도 하지 않는다.
    """
    driven = {str(robot_id) for robot_id in driven_robot_ids}
    total = 0
    for profile in ROBOT_RUNTIME_PROFILES.values():
        if profile.robot_id in driven:
            continue
        root = stage.GetPrimAtPath(profile.xform_root_path)
        if not root.IsValid():
            continue
        held = 0
        with session_edit(stage):
            for prim in Usd.PrimRange(root):
                type_name = str(prim.GetTypeName())
                if "Revolute" in type_name:
                    axis = "angular"
                elif "Prismatic" in type_name:
                    axis = "linear"
                else:
                    continue
                state_attr = prim.GetAttribute(f"state:{axis}:physics:position")
                if not state_attr or state_attr.Get() is None:
                    continue
                current = float(state_attr.Get())
                drive = UsdPhysics.DriveAPI.Apply(prim, axis)
                drive.CreateTargetPositionAttr().Set(current)
                if drive.GetStiffnessAttr().Get() in (None, 0.0):
                    drive.CreateStiffnessAttr().Set(
                        ARM_DRIVE_STIFFNESS if axis == "angular" else 1.0e6
                    )
                    drive.CreateDampingAttr().Set(
                        ARM_DRIVE_DAMPING if axis == "angular" else 1.0e5
                    )
                held += 1
        if held:
            print(f"   방치 로봇 고정: {profile.robot_id} joint {held}개")
        total += held
    return total


def configure_contact_colliders(stage, profile):
    """접촉 보고가 필요한 prim에 ContactReportAPI를 적용한다."""
    applied = 0
    targets = (profile.gripper_root_path, profile.tree_root_path)
    with session_edit(stage):
        for path in targets:
            root = stage.GetPrimAtPath(path)
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                PhysxSchema.PhysxContactReportAPI.Apply(prim)
                applied += 1
    return applied


# ══════════════════════════════════════════════════════════════
# Planning proxy 추출
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProxySpec:
    """PlanningScene 으로 발행할 정적 나무 proxy 하나."""

    obstacle_id: str
    shape: int  # 1=sphere, 2=box, 3=capsule
    obstacle_class: int  # 1=trunk, 2=branch
    position: np.ndarray
    orientation_xyzw: np.ndarray
    dimensions: np.ndarray
    safety_margin: float


def _voxel_downsample(points, voxel_size):
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def _cross_section_radius(points, center, search_radius):
    """center 주변 구간의 실제 단면 반경을 잰다.

    표면 mesh 에서 SVD 특이값을 그대로 반지름으로 쓰면 안 된다. 굵은
    몸통이라도 좁은 이웃만 보면 표면 조각이 거의 평평해서 두 번째·세
    번째 특이값이 0 에 가깝게 나오고, 그러면 몸통이 가는 가지로 분류된다.

    대신 첫 주성분을 그 구간의 장축으로 보고, 그 축에서 떨어진 수직
    거리의 중앙값을 반지름으로 쓴다. 이게 문서가 말하는 로컬 PCA 반경이다.
    """
    offsets = points - center
    mask = np.linalg.norm(offsets, axis=1) <= search_radius
    local = offsets[mask]
    if local.shape[0] < 6:
        return 0.0
    centered = local - local.mean(axis=0)
    try:
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    axis = basis[0]
    along = centered @ axis
    perpendicular = centered - np.outer(along, axis)
    return float(np.median(np.linalg.norm(perpendicular, axis=1)))


def tree_world_bounds(stage, tree_root_path, margin=0.20):
    """담당 나무의 world AABB. 어느 나무의 장애물인지 가르는 데 쓴다."""
    prim = stage.GetPrimAtPath(tree_root_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    low = np.array([box.GetMin()[i] for i in range(3)], dtype=float) - margin
    high = np.array([box.GetMax()[i] for i in range(3)], dtype=float) + margin
    return low, high


def _authored_tree_collider_specs(stage, tree_root_path):
    """stage 에 이미 있는 PhysX 나무 collider 를 planning proxy 로 읽는다.

    계획 proxy 는 로봇이 실제로 부딪히는 형상과 같아야 한다. collider 가
    authored 되어 있으면 mesh 에서 다시 추정하지 않고 그대로 쓰는 편이,
    RRT/RMPflow 가 보는 세계와 PhysX 가 보는 세계를 어긋나지 않게 한다.

    collider root 는 전역 경로 하나인데 씬에는 나무가 둘이다. 담당 나무의
    AABB 밖에 있는 collider 는 다른 나무 것이므로 버린다. 이 필터가 없으면
    robot_02 가 4 m 떨어진 나무 1 의 장애물로 계획해, 정작 자기 나무는
    장애물이 하나도 없는 상태로 팔을 넣는다.
    """
    root = stage.GetPrimAtPath(RUNTIME_TREE_COLLIDER_ROOT_PATH)
    if not root.IsValid():
        return ()
    bounds = tree_world_bounds(stage, tree_root_path)
    specs = []
    for child in root.GetChildren():
        capsule = UsdGeom.Capsule(child)
        if not capsule:
            continue
        radius = capsule.GetRadiusAttr().Get()
        if radius is None:
            continue
        radius = float(radius)
        height = float(capsule.GetHeightAttr().Get() or 0.0)
        if radius < TREE_MIN_PCA_RADIUS_M * 0.5:
            continue
        position, rotation = get_prim_world_pose(stage, str(child.GetPath()))
        if bounds is not None and (
            np.any(position < bounds[0]) or np.any(position > bounds[1])
        ):
            continue
        is_trunk = radius >= TREE_MIN_PCA_RADIUS_M * 2.0
        specs.append(
            ProxySpec(
                obstacle_id=child.GetName(),
                shape=3,  # SHAPE_CAPSULE
                obstacle_class=1 if is_trunk else 2,
                position=position,
                orientation_xyzw=quat_wxyz_to_xyzw(rot_matrix_to_quat(rotation)),
                dimensions=np.array([radius, height, 0.0], dtype=float),
                safety_margin=ROBOT_TREE_SAFETY_MARGIN_M,
            )
        )
    return tuple(specs)


def _limit_proxies(stage, profile, specs):
    """proxy 수를 제한한다. 개수를 줄여도 안전거리 자체는 축소하지 않는다."""
    if len(specs) <= MAX_BRANCH_PROXIES:
        return tuple(specs)
    base_position, _ = get_prim_world_pose(stage, profile.base_path)
    specs.sort(key=lambda spec: float(np.linalg.norm(spec.position - base_position)))
    return tuple(specs[:MAX_BRANCH_PROXIES])


def extract_static_planning_proxy_specs(stage, profile):
    """RRT/RMPflow 용 정적 나무 proxy 목록을 만든다.

    authored PhysX collider 가 있으면 그것을 그대로 쓴다. 없을 때만 visual
    mesh 에서 추정한다. 문서 규약대로 연결 성분을 40 mm 구간으로 나누고
    로컬 PCA 반경이 20 mm 이상인 구간만 obstacle 로 삼으며, 그보다 가는
    구간과 잎은 PhysX 와 planning 양쪽에서 제외한다.
    """
    tree_path = profile.tree_root_path
    authored = _authored_tree_collider_specs(stage, tree_path)
    if authored:
        print(
            f"   [{profile.robot_id}] authored PhysX collider "
            f"{len(authored)}개를 proxy 로 사용"
        )
        return _limit_proxies(stage, profile, list(authored))
    print(f"   [{profile.robot_id}] 구조 mesh 에서 proxy 를 추정합니다.")

    points = prim_world_points(
        stage, tree_path, 40000, TREE_NON_STRUCTURAL_KEYWORDS
    )
    if points.shape[0] == 0:
        return ()
    samples = _voxel_downsample(points, TREE_SEGMENT_LENGTH_M)
    specs = []
    trunk_z = float(np.percentile(points[:, 2], 35.0))
    # 굵은 몸통의 단면을 덮으려면 이웃이 지름보다 커야 한다.
    search_radius = TREE_SEGMENT_LENGTH_M * 3.0
    for index, center in enumerate(samples):
        radius = _cross_section_radius(points, center, search_radius)
        if radius < TREE_MIN_PCA_RADIUS_M:
            continue
        is_trunk = center[2] <= trunk_z
        specs.append(
            ProxySpec(
                obstacle_id=f"{'trunk' if is_trunk else 'branch'}_{index:04d}",
                shape=1,  # SHAPE_SPHERE
                obstacle_class=1 if is_trunk else 2,
                position=np.asarray(center, dtype=float),
                orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                dimensions=np.array(
                    [max(radius, PROXY_DEFAULT_SPHERE_RADIUS_M), 0.0, 0.0]
                ),
                safety_margin=ROBOT_TREE_SAFETY_MARGIN_M,
            )
        )
    return _limit_proxies(stage, profile, specs)


def visualize_planning_proxies(stage, world, profile, specs):
    """계획 proxy 를 화면에 그린다. 계획 자체는 표시와 무관하게 동작한다."""
    if not SHOW_PLANNING_DEBUG or not specs:
        return
    group = f"{PLANNING_OBSTACLE_ROOT_PATH}/{profile.robot_id}"
    with session_edit(stage):
        if not stage.GetPrimAtPath(PLANNING_OBSTACLE_ROOT_PATH).IsValid():
            UsdGeom.Xform.Define(stage, PLANNING_OBSTACLE_ROOT_PATH)
        if not stage.GetPrimAtPath(group).IsValid():
            UsdGeom.Xform.Define(stage, group)
    for index, spec in enumerate(specs[:MAX_BRANCH_PROXIES]):
        path = f"{group}/proxy_{index:03d}"
        if stage.GetPrimAtPath(path).IsValid():
            continue
        try:
            world.scene.add(
                VisualSphere(
                    prim_path=path,
                    name=f"{profile.robot_id}_proxy_{index:03d}",
                    position=spec.position,
                    radius=float(spec.dimensions[0]),
                    color=np.array([1.0, 0.85, 0.1]),
                )
            )
        except Exception:  # noqa: BLE001 - 시각화 실패는 실행을 막지 않는다
            break


def proxy_clearance(point, specs):
    """점에서 가장 가까운 proxy 표면까지의 거리(안전거리 포함)."""
    point = np.asarray(point, dtype=float)
    if not specs:
        return math.inf, ""
    best = math.inf
    closest = ""
    for spec in specs:
        surface = (
            float(np.linalg.norm(point - spec.position))
            - float(spec.dimensions[0])
            - spec.safety_margin
        )
        if surface < best:
            best = surface
            closest = spec.obstacle_id
    return best, closest


# ══════════════════════════════════════════════════════════════
# 시간 매개화 궤적
# ══════════════════════════════════════════════════════════════
def create_trajectory_generator():
    """Lula c-space trajectory generator. 실패하면 None 을 준다."""
    try:
        generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=str(DESCRIPTION_PATH),
            urdf_path=str(URDF_PATH),
        )
    except Exception as error:  # noqa: BLE001
        print(f"   [WARN] trajectory generator 초기화 실패: {error}")
        return None
    for setter, limits in (
        ("set_c_space_velocity_limits", RRT_TRAJECTORY_VELOCITY_LIMITS),
        ("set_c_space_acceleration_limits", RRT_TRAJECTORY_ACCELERATION_LIMITS),
        ("set_c_space_jerk_limits", RRT_TRAJECTORY_JERK_LIMITS),
    ):
        # 버전에 따라 setter 이름이 다르다. 없으면 기본 제한을 쓴다.
        if hasattr(generator, setter):
            try:
                getattr(generator, setter)(limits)
            except Exception:  # noqa: BLE001
                pass
    return generator


def sample_trajectory(generator, joint_path):
    """관절 경로를 시간 매개화해 (times, positions, velocities) 로 표본화한다.

    RRT waypoint 를 그대로 선형 보간해 실행하지 않는다는 문서 규약을 지키기
    위한 단계다. 생성기가 없으면 속도 제한을 지키는 등속 표본으로 대체한다.
    """
    joint_path = np.asarray(joint_path, dtype=float)
    if joint_path.shape[0] < 2:
        return np.zeros(1), joint_path, np.zeros_like(joint_path)
    if generator is not None:
        try:
            trajectory = generator.compute_c_space_trajectory(joint_path)
            if trajectory is not None:
                start = float(trajectory.start_time)
                end = float(trajectory.end_time)
                count = max(
                    2, int(math.ceil((end - start) / RRT_TRAJECTORY_SAMPLE_DT_S))
                )
                times = np.linspace(start, end, count)
                positions = np.array(
                    [trajectory.get_joint_targets(t)[0] for t in times], dtype=float
                )
                velocities = np.array(
                    [trajectory.get_joint_targets(t)[1] for t in times], dtype=float
                )
                return times - start, positions, velocities
        except Exception as error:  # noqa: BLE001
            print(f"   [WARN] trajectory 변환 실패, 등속 표본으로 대체: {error}")
    segments = []
    for index in range(joint_path.shape[0] - 1):
        delta = joint_path[index + 1] - joint_path[index]
        span = float(np.max(np.abs(delta) / RRT_TRAJECTORY_VELOCITY_LIMITS))
        steps = max(2, int(math.ceil(span / RRT_TRAJECTORY_SAMPLE_DT_S)))
        for step in range(steps):
            segments.append(joint_path[index] + delta * (step / float(steps)))
    segments.append(joint_path[-1])
    positions = np.array(segments, dtype=float)
    times = np.arange(positions.shape[0]) * RRT_TRAJECTORY_SAMPLE_DT_S
    velocities = np.gradient(positions, RRT_TRAJECTORY_SAMPLE_DT_S, axis=0)
    return times, positions, velocities


# ══════════════════════════════════════════════════════════════
# 접촉 감시
# ══════════════════════════════════════════════════════════════
class ContactMonitorBase:
    """PhysX Contact Report 를 구독하는 감시자의 공통부."""

    def __init__(self, stage):
        self.stage = stage
        self.triggered = False
        self.detail = ""
        self._subscription = None

    def _match(self, path_a, path_b):
        raise NotImplementedError

    def start(self):
        physx = omni.physx.get_physx_simulation_interface()
        self._subscription = physx.subscribe_contact_report_events(self._on_contact)

    def stop(self):
        self._subscription = None

    def reset(self):
        self.triggered = False
        self.detail = ""

    def _on_contact(self, contact_headers, contact_data):
        for header in contact_headers:
            if header.type != SimulationEvent.CONTACT_FOUND:
                continue
            path_a = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            path_b = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            if self._match(path_a, path_b):
                self.triggered = True
                self.detail = f"{path_a} <-> {path_b}"
                return


class RobotTreeContactMonitor(ContactMonitorBase):
    """로봇과 나무의 실제 PhysX 접촉을 감지한다.

    사과-그리퍼의 의도된 접촉과 로봇 자체 접촉은 조건에 포함하지 않는다.
    """

    def __init__(self, stage, robot_root, tree_root):
        super().__init__(stage)
        self.robot_root = robot_root
        self.tree_root = tree_root

    def _match(self, path_a, path_b):
        robot_hit = path_a.startswith(self.robot_root) or path_b.startswith(
            self.robot_root
        )
        if not robot_hit:
            return False
        tree_path = path_b if path_a.startswith(self.robot_root) else path_a
        if not tree_path.startswith(self.tree_root):
            return False
        # 잎에 스치는 것은 실패가 아니다. 몸통과 가지 접촉만 중단 사유다.
        lowered = tree_path.lower()
        if any(keyword in lowered for keyword in TREE_NON_STRUCTURAL_KEYWORDS):
            return False
        return True


class RobotAppleContactMonitor(ContactMonitorBase):
    """palm 이 사과에 먼저 닿았는지 확인한다.

    palm 보다 손가락이 먼저 닿으면 포위 파지가 아니라 끝단 밀기가 된다.
    """

    def __init__(self, stage, apple_path, palm_path, gripper_root):
        super().__init__(stage)
        self.apple_path = apple_path
        self.palm_path = palm_path
        self.gripper_root = gripper_root
        self.palm_contact = False
        self.finger_contact = False

    def reset(self):
        super().reset()
        self.palm_contact = False
        self.finger_contact = False

    def _match(self, path_a, path_b):
        if self.apple_path not in path_a and self.apple_path not in path_b:
            return False
        joined = f"{path_a} {path_b}"
        if self.palm_path in joined:
            self.palm_contact = True
        elif self.gripper_root in joined:
            self.finger_contact = True
        return False  # 감시만 하고 자체로 중단시키지 않는다


class JointBreakRecorder:
    """PhysX 가 보고한 joint 파손을 모아 둔다. 프로세스에 하나만 둔다.

    파손 판정을 USD 의 ``physics:jointEnabled`` 로 하면 안 된다. PhysX 는
    실행 중에 joint 를 끊어도 그 결과를 USD 속성에 되쓰지 않는다. 그래서
    화면에서는 사과가 분명히 떨어졌는데도 속성은 계속 True 로 남고,
    PULL 이 항상 ``STEM_NOT_BROKEN`` 으로 실패했다.

    PhysX 는 대신 ``SimulationEvent.JOINT_BREAK`` 를 발행한다. 그것이
    파손의 유일한 신뢰할 수 있는 신호다.
    """

    def __init__(self):
        self._paths = set()
        self._subscription = None

    def start(self):
        if self._subscription is not None:
            return
        stream = omni.physx.get_physx_interface().get_simulation_event_stream_v2()
        self._subscription = stream.create_subscription_to_pop(self._on_event)

    def stop(self):
        self._subscription = None

    def _on_event(self, event):
        if event.type != int(SimulationEvent.JOINT_BREAK):
            return
        encoded = event.payload["jointPath"]
        path = PhysicsSchemaTools.decodeSdfPath(encoded[0], encoded[1])
        self._paths.add(str(path))
        print(f"   [PHYSX] joint 파손: {path}")

    def is_broken(self, joint_path):
        return str(joint_path) in self._paths

    def clear(self):
        """Timeline reset 처럼 물리를 다시 세울 때 기록을 버린다."""
        self._paths.clear()


JOINT_BREAK_RECORDER = JointBreakRecorder()


class JointBreakMonitor:
    """stem FixedJoint 가 실제로 끊어졌는지 확인한다."""

    def __init__(self, stage, joint_path):
        self.stage = stage
        self.joint_path = joint_path

    def is_broken(self):
        if not self.joint_path:
            return False
        # PhysX 이벤트가 1순위 근거다.
        if JOINT_BREAK_RECORDER.is_broken(self.joint_path):
            return True
        prim = self.stage.GetPrimAtPath(self.joint_path)
        if not prim.IsValid():
            # joint prim 자체가 사라졌으면 끊어진 것으로 본다.
            return True
        joint = UsdPhysics.Joint(prim)
        if not joint:
            return True
        # USD 속성은 사용자가 직접 껐을 때만 의미가 있다. PhysX 파손은
        # 여기에 반영되지 않으므로 보조 근거로만 쓴다.
        return joint.GetJointEnabledAttr().Get() is False


def find_robot_tree_physx_overlap(stage, robot_root, tree_root):
    """실행 직전에 로봇과 나무 collider 가 이미 겹쳐 있는지 검사한다.

    잎은 collision 대상이 아니므로 겹침 판정에서도 뺀다. 넣으면 사과에
    다가가기만 해도 잎에 닿아 실행 전 검사에서 항상 걸린다.
    """
    robot_points = prim_world_points(stage, robot_root, 4000)
    tree_points = prim_world_points(
        stage, tree_root, 4000, TREE_NON_STRUCTURAL_KEYWORDS
    )
    if robot_points.shape[0] == 0 or tree_points.shape[0] == 0:
        return None
    sample = tree_points[:: max(1, tree_points.shape[0] // 500)]
    best = math.inf
    best_point = None
    for point in sample:
        distance = float(np.min(np.linalg.norm(robot_points - point, axis=1)))
        if distance < best:
            best = distance
            best_point = point
    if best < 0.005:
        return {"distance": best, "point": best_point}
    return None


# ══════════════════════════════════════════════════════════════
# 로봇 런타임
# ══════════════════════════════════════════════════════════════
class RobotRuntime:
    """한 로봇이 소유하는 실행 상태 전부.

    한 프로세스가 이 객체를 두 개 만들면 하나의 Isaac Sim World 안에서 두
    로봇이 각자 자기 나무의 사과를 딴다. 로봇에 딸린 것은 전부 여기 있고
    모듈 전역에는 남기지 않는다.
    """

    def __init__(self, world, stage, profile):
        self.world = world
        self.stage = stage
        self.profile = profile
        self.robot_id = profile.robot_id

        require_prim(stage, profile.articulation_root_joint_path)
        require_prim(stage, profile.robot_prim_path)

        self.assemblies = discover_apple_assemblies(stage, profile)
        self.stem_joint_paths = tuple(
            item["joint"] for item in self.assemblies if item["joint"]
        )
        for joint_path in self.stem_joint_paths:
            configure_breakable_joint(
                stage, joint_path, STEM_BREAK_FORCE_N, STEM_BREAK_TORQUE_NM
            )
        print(
            f"   [{self.robot_id}] 사과 assembly {len(self.assemblies)}개, "
            f"stem joint {len(self.stem_joint_paths)}개 "
            f"({STEM_BREAK_FORCE_N:.0f} N / {STEM_BREAK_TORQUE_NM:.2f} N·m)"
        )

        configure_joint_drives(stage, profile)
        configure_contact_colliders(stage, profile)

        self.articulation = world.scene.add(
            SingleManipulator(
                prim_path=profile.articulation_prim_path,
                name=f"m0617_{profile.robot_id}",
                end_effector_prim_path=profile.palm_path,
            )
        )
        self.indices = {}
        self.solver = None
        self.motion = None
        self.proxies = ()
        self.tree_signature = tree_scene_signature(stage, profile.tree_root_path)

    # -- 초기화 ----------------------------------------------------------
    def after_world_reset(self):
        """World.reset() 뒤에 관절 인덱스와 solver 를 준비한다."""
        self.indices = {
            name: index for index, name in enumerate(self.articulation.dof_names)
        }
        self.apply_arm_target(self.profile.initial_arm_joints_rad)
        self.apply_gripper_target(GRIPPER_OPEN, GRIPPER_ENTRY_MAX_FORCE)
        self.solver = self._create_ik_solver()
        self.proxies = extract_static_planning_proxy_specs(self.stage, self.profile)
        visualize_planning_proxies(
            self.stage, self.world, self.profile, self.proxies
        )
        self.motion = CollisionAwareMotion(self)
        return self

    def _create_ik_solver(self):
        for path in (DESCRIPTION_PATH, URDF_PATH):
            if not path.exists():
                raise HarvestError(f"Lula 입력 파일을 찾을 수 없습니다: {path}")
        solver = LulaKinematicsSolver(
            robot_description_path=str(DESCRIPTION_PATH),
            urdf_path=str(URDF_PATH),
        )
        base_position, base_rotation = self.base_pose()
        solver.set_robot_base_pose(base_position, rot_matrix_to_quat(base_rotation))
        return solver

    def close(self):
        if self.motion is not None:
            self.motion.close()

    # -- pose ------------------------------------------------------------
    def base_pose(self):
        return get_prim_world_pose(self.stage, self.profile.base_path)

    def current_tcp_pose(self):
        """물리 수확 TCP의 (position, rotation_matrix).

        palm frame 에서 palm 로컬 +Y 로 0.0908 m 떨어진 점이다. URDF 의 보조
        gripper_frame 을 쓰지 않는 이유는 USD 조립 자세와 RPY 가 달라 실제
        TCP 와 Lula 목표가 서로 다른 위치로 수렴할 수 있기 때문이다.
        """
        position, rotation = get_prim_world_pose(self.stage, self.profile.palm_path)
        return position + rotation[:, 1] * PALM_TO_TCP_Y_M, rotation

    def link6_target_from_tcp(self, tcp_position, tcp_rotation):
        """TCP 목표를 Lula 제어 frame(link_6) 목표로 변환한다."""
        palm_position, palm_rotation = get_prim_world_pose(
            self.stage, self.profile.palm_path
        )
        link6_position, link6_rotation = get_prim_world_pose(
            self.stage, self.profile.link6_path
        )
        relative_rotation = link6_rotation.T @ palm_rotation
        relative_offset = link6_rotation.T @ (palm_position - link6_position)
        tcp_rotation = np.asarray(tcp_rotation, dtype=float)
        goal_rotation = tcp_rotation @ relative_rotation.T
        palm_goal = (
            np.asarray(tcp_position, dtype=float)
            - tcp_rotation[:, 1] * PALM_TO_TCP_Y_M
        )
        return palm_goal - goal_rotation @ relative_offset, goal_rotation

    # -- 관절 ------------------------------------------------------------
    def apply_arm_target(self, joint_positions):
        joint_positions = np.asarray(joint_positions, dtype=float)
        command = np.full(self.articulation.num_dof, np.nan)
        for name, value in zip(ARM_JOINTS, joint_positions):
            if name in self.indices:
                command[self.indices[name]] = value
        self.articulation.apply_action(ArticulationAction(joint_positions=command))

    def apply_gripper_target(self, positions, max_force=None):
        positions = np.asarray(positions, dtype=float)
        if positions.shape[0] != len(GRIPPER_JOINTS):
            raise ValueError(
                f"그리퍼 목표는 {len(GRIPPER_JOINTS)}개여야 합니다: {positions.shape}"
            )
        if max_force is not None:
            set_gripper_drive_max_force(self.stage, self.profile, max_force)
        command = np.full(self.articulation.num_dof, np.nan)
        for name, value in zip(GRIPPER_JOINTS, positions):
            if name in self.indices:
                command[self.indices[name]] = value
        self.articulation.apply_action(ArticulationAction(joint_positions=command))

    def read_arm_joints(self):
        positions = self.articulation.get_joint_positions()
        return np.array(
            [positions[self.indices[name]] for name in ARM_JOINTS], dtype=float
        )

    # -- IK --------------------------------------------------------------
    def solve_ik(self, tcp_position, tcp_rotation, warm_start=None):
        goal_position, goal_rotation = self.link6_target_from_tcp(
            tcp_position, tcp_rotation
        )
        base_position, base_rotation = self.base_pose()
        self.solver.set_robot_base_pose(
            base_position, rot_matrix_to_quat(base_rotation)
        )
        if warm_start is None:
            warm_start = self.read_arm_joints()
        solution, success = self.solver.compute_inverse_kinematics(
            frame_name="link_6",
            warm_start=np.asarray(warm_start, dtype=float),
            target_position=goal_position,
            target_orientation=rot_matrix_to_quat(goal_rotation),
        )
        if not success:
            return None
        # LulaKinematicsSolver 는 (joint_positions, success) 를 준다.
        # ArticulationKinematicsSolver 쪽은 ArticulationAction 을 주므로,
        # 어느 쪽이 와도 관절 배열을 꺼낼 수 있게 둔다.
        joints = getattr(solution, "joint_positions", solution)
        if joints is None:
            return None
        return np.asarray(joints, dtype=float)[: len(ARM_JOINTS)]

    def validate_planned_ik(self, waypoints):
        """계획 waypoint 전부에 IK 해가 있는지 실행 전에 확인한다.

        이동을 시작한 뒤에야 도달 불가를 발견하면 로봇이 나무 안에서 멈춘다.
        """
        joints = self.read_arm_joints()
        solutions = []
        for waypoint in waypoints:
            rotation = quat_to_rot_matrix(
                quat_xyzw_to_wxyz(waypoint.orientation_xyzw)
            )
            solution = self.solve_ik(waypoint.position, rotation, joints)
            if solution is None:
                raise ApproachUnreachableError(
                    f"{waypoint.phase} waypoint의 IK 해를 찾지 못했습니다."
                )
            solutions.append(solution)
            joints = solution
        return solutions

    # -- 사과 ------------------------------------------------------------
    def apple_center(self, apple_path):
        return compute_live_prim_center(self.stage, apple_path)

    def nearest_assembly(self, world_position):
        """target 중심과 가장 가까운 사과 assembly."""
        if not self.assemblies:
            raise HarvestError(f"{self.robot_id}: 사과 assembly 를 찾지 못했습니다.")
        world_position = np.asarray(world_position, dtype=float)
        best = None
        for assembly in self.assemblies:
            center = self.apple_center(assembly["apple"])
            distance = float(np.linalg.norm(center - world_position))
            if best is None or distance < best[0]:
                best = (distance, assembly, center)
        return best[1], best[2]

    def refresh_scene_if_moved(self):
        """나무가 실제로 옮겨졌으면 proxy 를 다시 만든다."""
        signature = tree_scene_signature(self.stage, self.profile.tree_root_path)
        if signature == self.tree_signature:
            return False
        self.tree_signature = signature
        self.proxies = extract_static_planning_proxy_specs(self.stage, self.profile)
        return True

    # -- 그리퍼 진입 여유 --------------------------------------------------
    def entry_swept_clearance(self, apple_center, tcp_position=None):
        """TCP -> 사과 중심 선분을 손가락 collider 로 sweep 해 실제 여유를 잰다.

        관절각과 개구폭을 선형으로 환산하지 않고 authored collision mesh 를
        직접 쓴다. 반환값은 사과 반지름을 제외한 여유다.
        """
        apple_center = np.asarray(apple_center, dtype=float)
        if tcp_position is None:
            tcp_position, _ = self.current_tcp_pose()
        tcp_position = np.asarray(tcp_position, dtype=float)
        points = prim_world_points(self.stage, self.profile.gripper_root_path, 8000)
        if points.shape[0] == 0:
            return math.inf
        distances = np.array(
            [
                point_to_line_distance(point, tcp_position, apple_center)
                for point in points
            ]
        )
        return float(np.min(distances)) - APPLE_RADIUS_M

    def select_entry_preshape(self, apple_center):
        """안전 여유를 만족하면서 목표 개구에 가장 가까운 진입 자세.

        명목 지름 80 mm 사과에서 손가락 안쪽 면이 중심에서 양쪽 50 mm 가
        되도록, 면당 10 mm 여유에 가장 가까운 후보를 고른다.
        """
        best = None
        for name, preshape in GRIPPER_ENTRY_CANDIDATES:
            self.apply_gripper_target(preshape, GRIPPER_ENTRY_MAX_FORCE)
            for _ in range(ENTRY_PRESHAPE_SAMPLE_STEPS):
                self.world.step(render=False)
            clearance = self.entry_swept_clearance(apple_center)
            if clearance < ENTRY_SWEEP_MIN_CLEARANCE_M:
                continue
            error = abs(clearance + APPLE_RADIUS_M - ENTRY_TARGET_HALF_OPENING_M)
            if best is None or error < best[0]:
                best = (error, name, preshape, clearance)
        if best is None:
            raise CollisionRiskError(
                f"{self.robot_id}: 안전 여유를 만족하는 그리퍼 진입 자세를 찾지 "
                f"못했습니다. 요구 clearance "
                f"{ENTRY_SWEEP_MIN_CLEARANCE_M * 1000:.1f} mm"
            )
        _, name, preshape, clearance = best
        print(
            f"   [{self.robot_id}] 진입 자세 {name}, "
            f"swept clearance {clearance * 1000:.1f} mm"
        )
        return preshape


# ══════════════════════════════════════════════════════════════
# 충돌 인지 이동
# ══════════════════════════════════════════════════════════════
class CollisionAwareMotion:
    """RMPflow 로 목표를 추종하면서 접촉·정체를 감시하는 실행 계층."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.world = runtime.world
        self.stage = runtime.stage
        self.profile = runtime.profile
        self.tree_monitor = RobotTreeContactMonitor(
            runtime.stage,
            runtime.profile.robot_prim_path,
            runtime.profile.tree_root_path,
        )
        self.tree_monitor.start()
        self.trajectory_generator = create_trajectory_generator()
        self.rmpflow = None
        self.articulation_policy = None
        if RMPFLOW_CONFIG_PATH.exists():
            try:
                self.rmpflow = RmpFlow(
                    robot_description_path=str(DESCRIPTION_PATH),
                    urdf_path=str(URDF_PATH),
                    rmpflow_config_path=str(RMPFLOW_CONFIG_PATH),
                    end_effector_frame_name="link_6",
                    maximum_substep_size=RMPFLOW_MAXIMUM_SUBSTEP_S,
                )
                self.articulation_policy = ArticulationMotionPolicy(
                    runtime.articulation, self.rmpflow, RRT_TRAJECTORY_SAMPLE_DT_S
                )
            except Exception as error:  # noqa: BLE001
                print(
                    f"   [WARN] {runtime.robot_id} RMPflow 초기화 실패, "
                    f"관절 보간으로 실행합니다: {error}"
                )
                self.rmpflow = None

    def close(self):
        self.tree_monitor.stop()

    def _check_contact(self):
        if self.tree_monitor.triggered:
            detail = self.tree_monitor.detail
            self.tree_monitor.reset()
            raise UnexpectedContactError(f"로봇-나무 접촉이 감지되었습니다: {detail}")

    def move_via_joint_trajectory(self, tcp_position, tcp_rotation, label="TRANSIT"):
        """목표에서 IK 를 풀고 관절공간 시간매개화 궤적으로 이동한다.

        RMPflow 는 목표 근처에서 자세를 다듬는 반응형 계층이라, 나무에서
        컨베이어까지처럼 크게 선회하는 구간에서는 수렴하지 못한다. 실측에서
        robot_02 는 1080 step 뒤에도 813 mm 가 남았다. 컨베이어가 나무
        반대편이라 팔이 거의 반바퀴를 돌아야 하기 때문이다.

        그 구간은 목표 관절각을 먼저 구하고 속도·가속도 제약을 반영한
        궤적으로 따라가는 편이 확실하다. 문서의 trajectory generation 단계가
        이것이다.
        """
        runtime = self.runtime
        start = runtime.read_arm_joints()
        goal = runtime.solve_ik(tcp_position, tcp_rotation, start)
        if goal is None:
            raise ApproachUnreachableError(f"{label}: 목표의 IK 해가 없습니다.")

        _times, positions, _velocities = sample_trajectory(
            self.trajectory_generator, np.vstack([start, goal])
        )
        for joints in positions:
            runtime.apply_arm_target(joints)
            self.world.step(render=True)
            self._check_contact()

        # 궤적 끝에서 드라이브가 목표를 따라잡을 시간을 준다.
        for _ in range(GRASP_SETTLE_STEPS):
            runtime.apply_arm_target(goal)
            self.world.step(render=True)
            self._check_contact()

        position, rotation = runtime.current_tcp_pose()
        error = float(np.linalg.norm(position - np.asarray(tcp_position, dtype=float)))
        print(f"   [{runtime.robot_id}] {label} 도착, 위치 오차 {error * 1000:.1f} mm")
        return True

    def move_to_pose(
        self,
        tcp_position,
        tcp_rotation,
        *,
        max_steps=RMPFLOW_SEGMENT_STEPS,
        label="MOVE",
        on_step=None,
    ):
        """TCP 목표까지 이동한다. 도달하면 True."""
        runtime = self.runtime
        tcp_position = np.asarray(tcp_position, dtype=float)
        tcp_rotation = np.asarray(tcp_rotation, dtype=float)

        if self.rmpflow is not None:
            goal_position, goal_rotation = runtime.link6_target_from_tcp(
                tcp_position, tcp_rotation
            )
            base_position, base_rotation = runtime.base_pose()
            self.rmpflow.set_robot_base_pose(
                base_position, rot_matrix_to_quat(base_rotation)
            )
            self.rmpflow.set_end_effector_target(
                target_position=goal_position,
                target_orientation=rot_matrix_to_quat(goal_rotation),
            )
        else:
            solution = runtime.solve_ik(tcp_position, tcp_rotation)
            if solution is None:
                raise IkFailedError(f"{label}: IK 해를 찾지 못했습니다.")
            runtime.apply_arm_target(solution)

        last_position, last_rotation = runtime.current_tcp_pose()
        stall_steps = 0
        for step in range(max_steps):
            if self.rmpflow is not None:
                action = self.articulation_policy.get_next_articulation_action(
                    RRT_TRAJECTORY_SAMPLE_DT_S
                )
                runtime.articulation.apply_action(action)
            self.world.step(render=True)
            self._check_contact()
            if on_step is not None:
                on_step(step)

            position, rotation = runtime.current_tcp_pose()
            position_error = float(np.linalg.norm(position - tcp_position))
            rotation_error = rotation_error_deg(rotation, tcp_rotation)
            if (
                position_error <= TARGET_POSITION_TOLERANCE_M
                and rotation_error <= TARGET_ORIENTATION_TOLERANCE_DEG
            ):
                return True

            moved = float(np.linalg.norm(position - last_position))
            turned = rotation_error_deg(rotation, last_rotation)
            if (
                moved < RMPFLOW_STALL_POSITION_DELTA_M
                and turned < RMPFLOW_STALL_ROTATION_DELTA_DEG
            ):
                stall_steps += 1
            else:
                stall_steps = 0
                last_position, last_rotation = position, rotation
            if stall_steps >= RMPFLOW_STALL_STEPS:
                raise MotionTimeoutError(
                    f"{label}: {RMPFLOW_STALL_STEPS} step 동안 진전이 없습니다. "
                    f"위치 오차 {position_error * 1000:.1f} mm, "
                    f"자세 오차 {rotation_error:.1f} deg"
                )
        position, rotation = runtime.current_tcp_pose()
        raise MotionTimeoutError(
            f"{label}: {max_steps} step 안에 목표에 도달하지 못했습니다. "
            f"남은 위치 오차 {float(np.linalg.norm(position - tcp_position)) * 1000:.1f} mm, "
            f"자세 오차 {rotation_error_deg(rotation, tcp_rotation):.1f} deg"
        )


# ══════════════════════════════════════════════════════════════
# 수확 상태 머신
# ══════════════════════════════════════════════════════════════
class AppleHarvestFSM:
    """docs/features/harvesting.md 의 상태 흐름을 그대로 구현한다.

    각 단계는 독립적으로 호출할 수 있다. vision_apple_pick 의 Action
    서버가 motion_type 별로 하나씩 부른다.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.state = "TARGET_RECEIVED"
        self.apple_path = ""
        self.joint_path = ""
        self.approach_direction = vec(0.0, 0.0, 1.0)
        self.grasp_position = None
        self.grasp_rotation = None
        self.break_monitor = JointBreakMonitor(runtime.stage, "")
        self.apple_monitor = None

    def close(self):
        if self.apple_monitor is not None:
            self.apple_monitor.stop()

    def select_apple(self, world_position):
        """target 중심에 가장 가까운 사과로 문맥을 전환한다."""
        assembly, center = self.runtime.nearest_assembly(world_position)
        self.apple_path = assembly["apple"]
        self.joint_path = assembly["joint"] or ""
        self.break_monitor = JointBreakMonitor(self.runtime.stage, self.joint_path)
        if self.apple_monitor is not None:
            self.apple_monitor.stop()
        self.apple_monitor = RobotAppleContactMonitor(
            self.runtime.stage,
            self.apple_path,
            self.runtime.profile.palm_path,
            self.runtime.profile.gripper_root_path,
        )
        self.apple_monitor.start()
        return assembly, center

    # -- 단계 ------------------------------------------------------------
    def approach(self, apple_center, approach_direction=None):
        """staging -> pre-grasp -> 저속 진입까지."""
        self.state = "PRE_GRASP_PLANNING"
        runtime = self.runtime
        apple_center = np.asarray(apple_center, dtype=float)
        direction = (
            vec(0.0, 0.0, 1.0)
            if approach_direction is None
            else normalized(approach_direction)
        )
        self.approach_direction = direction
        rotation = make_approach_rotation_for_direction(direction)

        overlap = find_robot_tree_physx_overlap(
            runtime.stage,
            runtime.profile.robot_prim_path,
            runtime.profile.tree_root_path,
        )
        if overlap is not None:
            raise CollisionRiskError(
                f"실행 전 로봇-나무 collider 가 이미 겹쳐 있습니다: "
                f"{overlap['distance'] * 1000:.1f} mm"
            )

        # staging: 목표 사과를 obstacle 로 유지한 채 접근축 반대쪽 0.30 m.
        staging = apple_center - direction * STAGING_DISTANCE_M
        runtime.motion.move_to_pose(staging, rotation, label="STAGING")

        # pre-grasp: 목표 사과 obstacle 해제 후 같은 축으로 0.15 m.
        self.state = "APPROACH"
        pregrasp = apple_center - direction * PREGRASP_DISTANCE_M
        runtime.motion.move_to_pose(pregrasp, rotation, label="PRE_GRASP")

        preshape = runtime.select_entry_preshape(apple_center)
        if self.apple_monitor is not None:
            self.apple_monitor.reset()
        self._slow_entry(apple_center, direction, rotation, preshape)
        self.grasp_position, self.grasp_rotation = runtime.current_tcp_pose()
        return self.grasp_position, self.grasp_rotation

    def _slow_entry(self, apple_center, direction, rotation, preshape):
        """0.15 m 에서 0.03 m 까지, 그리고 마지막 0.03 m 를 더 낮은 속도로."""
        runtime = self.runtime
        start = apple_center - direction * PREGRASP_DISTANCE_M
        middle = apple_center - direction * FINAL_APPROACH_DISTANCE_M

        def guard(step):
            if step % ENTRY_LIVE_CHECK_INTERVAL_STEPS:
                return
            clearance = runtime.entry_swept_clearance(apple_center)
            if clearance < ENTRY_LIVE_MIN_CLEARANCE_M:
                raise CollisionRiskError(
                    f"진입 중 실측 여유가 부족합니다: {clearance * 1000:.1f} mm"
                )
            monitor = self.apple_monitor
            if monitor is not None and monitor.finger_contact and not monitor.palm_contact:
                raise CollisionRiskError(
                    "palm 보다 손가락 collider 가 먼저 사과에 접촉했습니다."
                )

        runtime.apply_gripper_target(preshape, GRIPPER_ENTRY_MAX_FORCE)
        self._interpolated_move(start, middle, rotation, RMPFLOW_SEGMENT_STEPS, guard)

        target = apple_center - direction * (APPLE_RADIUS_M * 0.5)

        def final_guard(step):
            guard(step)
            if self.apple_monitor is not None and self.apple_monitor.palm_contact:
                raise StopIteration

        try:
            self._interpolated_move(
                middle, target, rotation, RMPFLOW_SEGMENT_STEPS, final_guard
            )
        except StopIteration:
            print(
                f"   [{runtime.robot_id}] palm 접촉 확인. 현재 자세를 유지하고 "
                "GRASP 를 허용합니다."
            )

    def _interpolated_move(self, start, end, rotation, steps, on_step):
        """두 점 사이를 부드럽게 보간해 저속으로 이동한다."""
        runtime = self.runtime
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        for step in range(steps):
            alpha = smoothstep((step + 1) / float(steps))
            waypoint = start + (end - start) * alpha
            solution = runtime.solve_ik(waypoint, rotation)
            if solution is None:
                raise IkFailedError("진입 구간의 IK 해를 찾지 못했습니다.")
            runtime.apply_arm_target(solution)
            runtime.world.step(render=True)
            on_step(step)

    def grasp(self):
        """현재 pose 를 유지하고 그리퍼만 폐합한다."""
        self.state = "GRASP"
        runtime = self.runtime
        runtime.apply_gripper_target(GRIPPER_CLOSED, GRIPPER_GRASP_MAX_FORCE)
        for _ in range(GRASP_SETTLE_STEPS):
            runtime.world.step(render=True)
        # 운반 중 미끄러지지 않도록 유지 토크를 올린다.
        set_gripper_drive_max_force(
            runtime.stage, runtime.profile, GRIPPER_HOLD_MAX_FORCE
        )
        if self.apple_path:
            center = runtime.apple_center(self.apple_path)
            tcp, _ = runtime.current_tcp_pose()
            distance = float(np.linalg.norm(center - tcp))
            if distance > APPLE_GRASP_MAX_DISTANCE_M:
                raise HarvestError(
                    f"파지 후 사과가 TCP 에서 {distance * 1000:.1f} mm 떨어져 있습니다."
                )
        return True

    def twist(self):
        """손목만 45도를 1초 동안 회전한다."""
        self.state = "TWIST"
        runtime = self.runtime
        position, rotation = runtime.current_tcp_pose()
        total = math.radians(TWIST_ANGLE_DEG)
        for step in range(TWIST_STEPS):
            alpha = smoothstep((step + 1) / float(TWIST_STEPS))
            goal = rotation_about_axis(self.approach_direction, total * alpha) @ rotation
            solution = runtime.solve_ik(position, goal)
            if solution is None:
                raise IkFailedError("TWIST 구간의 IK 해를 찾지 못했습니다.")
            runtime.apply_arm_target(solution)
            runtime.world.step(render=True)
        return True

    def linear_pull(self):
        """줄기 반대 방향으로 50 mm/s, 최대 100 mm 를 일직선으로 당긴다."""
        self.state = "LINEAR_PULL"
        runtime = self.runtime
        start, rotation = runtime.current_tcp_pose()
        direction = -self.approach_direction
        physics_dt = runtime.world.get_physics_dt() or RRT_TRAJECTORY_SAMPLE_DT_S
        step_distance = PULL_SPEED_MPS * physics_dt
        steps = max(1, int(math.ceil(PULL_MAX_DISTANCE_M / step_distance)))
        last_progress = start
        stall_time = 0.0
        for step in range(steps):
            travelled = min(PULL_MAX_DISTANCE_M, step_distance * (step + 1))
            solution = runtime.solve_ik(start + direction * travelled, rotation)
            if solution is None:
                raise IkFailedError("PULL 구간의 IK 해를 찾지 못했습니다.")
            runtime.apply_arm_target(solution)
            runtime.world.step(render=True)
            if self.break_monitor.is_broken():
                print(
                    f"   [{runtime.robot_id}] stem 분리 확인: "
                    f"{travelled * 1000:.1f} mm 당김"
                )
                self.state = "STEM_BREAK_CHECK"
                return True
            position, _ = runtime.current_tcp_pose()
            if float(np.linalg.norm(position - last_progress)) > 0.001:
                last_progress = position
                stall_time = 0.0
            else:
                stall_time += physics_dt
                if stall_time >= MOTION_TIMEOUT_S:
                    raise MotionTimeoutError(
                        f"PULL: {MOTION_TIMEOUT_S:.0f} 초 동안 TCP 진전이 없습니다."
                    )
        # 최대 거리까지 당겼는데 끊어지지 않으면 성공이 아니다.
        if not self.break_monitor.is_broken():
            raise StemNotBrokenError(
                f"{PULL_MAX_DISTANCE_M * 1000:.0f} mm 를 당겼지만 stem 이 "
                "분리되지 않았습니다."
            )
        self.state = "STEM_BREAK_CHECK"
        return True

    def transport(self, position, rotation=None):
        """컨베이어 상공 경유점으로 이동한다."""
        self.state = "TRANSPORT"
        self.runtime.motion.move_via_joint_trajectory(
            position,
            self.grasp_rotation if rotation is None else rotation,
            label="TRANSPORT",
        )
        return True

    def place(self, position, rotation=None):
        """컨베이어 1 상면 30 mm 이하까지 내린다. 그리퍼는 열지 않는다."""
        self.state = "PLACE_ON_CONVEYOR"
        # TRANSPORT 와 같은 방식으로 내린다. 경유점에서 배치점까지는 0.15 m
        # 안팎이지만, 그 자세에서 RMPflow 로 직교 하강을 시키면 수렴하지
        # 못하고 253 mm 를 남긴 채 정체했다. 두 끝점의 IK 해를 잇는 궤적이
        # 확실하다.
        self.runtime.motion.move_via_joint_trajectory(
            position,
            self.grasp_rotation if rotation is None else rotation,
            label="PLACE",
        )
        return True

    def release(self):
        """현재 pose 를 유지하고 그리퍼만 개방한다."""
        self.state = "RELEASE"
        runtime = self.runtime
        runtime.apply_gripper_target(GRIPPER_OPEN, GRIPPER_HOLD_MAX_FORCE)
        for _ in range(GRASP_SETTLE_STEPS):
            runtime.world.step(render=True)
        return True

    def retract(self, position=None):
        """수직으로 이탈한 뒤 초기 자세로 돌아간다.

        상승량을 0.30 m 로 고정하면 안 된다. 배치 지점은 컨베이어 위라
        이미 팔 길이의 끝이고, 거기서 더 올리면 반경을 넘어 IK 해가 없어진다.
        실측에서 위치 오차 51.6 mm 로 정체해 timeout 이 났다. 그래서 닿는
        높이 중 가장 큰 값을 고르고, 어느 높이도 닿지 않으면 직교 상승을
        건너뛰고 관절 목표로 바로 복귀한다. 사과는 이미 놓은 뒤라 직교
        경로를 고집할 이유가 없다.
        """
        self.state = "RETRACT"
        runtime = self.runtime
        current, rotation = runtime.current_tcp_pose()

        if position is not None:
            runtime.motion.move_to_pose(
                np.asarray(position, dtype=float), rotation, label="RETRACT"
            )
        else:
            for lift in CONVEYOR_TRANSIT_HEIGHTS_M:
                if lift <= 0.0:
                    print(
                        f"   [{runtime.robot_id}] 직교 상승 여유가 없어 "
                        "관절 목표로 바로 복귀합니다."
                    )
                    break
                target = current + vec(0.0, 0.0, lift)
                if runtime.solve_ik(target, rotation) is None:
                    continue
                runtime.motion.move_to_pose(target, rotation, label="RETRACT")
                break

        runtime.apply_arm_target(runtime.profile.initial_arm_joints_rad)
        for _ in range(GRASP_SETTLE_STEPS):
            runtime.world.step(render=True)
        return True


# ══════════════════════════════════════════════════════════════
# 컨베이어 배치 지점
# ══════════════════════════════════════════════════════════════
def conveyor_place_pose(stage):
    """컨베이어 1 위 배치 목표 (position, rotation).

    벨트 상면에서 30 mm 이하로 낮춘 지점이다. 사과를 높은 곳에서 떨어뜨리지
    않고 벨트에 거의 닿은 상태에서 놓는다. 컨베이어는 한 대뿐이라 두 로봇이
    같은 지점을 쓰므로, 실제 배치 순서는 상위의 배치 mutex 가 정한다.
    """
    # 실제로 구동되는 벨트는 ConveyorBeltGraph 가 가리키는 Rollers 다.
    # 실측 bbox: x 2.85~3.21, y -1.26~0.72, 상면 z=0.769. 예전 배치 기준이던
    # 덮개 평면(y중심 0.586)은 벨트 맨 끝자락이라 사과가 마지막 롤러 틈에
    # 끼었고, 시각 롤러 에셋과 물리 벨트가 어긋나 있다는 사용자 관찰
    # (그림자 쪽에 떨어뜨리면 이동함)과 일치한다. 벨트 진행 방향(-Y) 기준
    # 카메라(y=-0.03) 앞쪽인 y=0.30 에 놓는다.
    plate = stage.GetPrimAtPath(RUNTIME_CONVEYOR_COLLIDER_PATH)
    if plate.IsValid():
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        box = cache.ComputeWorldBound(plate).ComputeAlignedRange()
        center_x = 0.5 * (float(box.GetMin()[0]) + float(box.GetMax()[0]))
        top_z = float(box.GetMax()[2])
        # y=0.30: 이송 방향(-Y) 기준 탑뷰 카메라(y=-0.03) 앞쪽.
        center = np.array([center_x, 0.30, top_z], dtype=float)
    elif stage.GetPrimAtPath(RUNTIME_CONVEYOR_COLLIDER_PATH).IsValid():
        prim = stage.GetPrimAtPath(RUNTIME_CONVEYOR_COLLIDER_PATH)
        center = compute_live_prim_center(stage, RUNTIME_CONVEYOR_COLLIDER_PATH)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        top_z = float(box.GetMax()[2]) if not box.IsEmpty() else float(center[2])
    else:
        center = compute_live_prim_center(stage, CONVEYOR_PATH)
        top_z = float(center[2])
    position = np.array(
        [center[0], center[1], top_z + PLACE_HEIGHT_ABOVE_BELT_M + APPLE_RADIUS_M],
        dtype=float,
    )
    # 아래로 향하는 파지 자세. 접근축이 -Z 이므로 palm +Y 가 아래를 본다.
    return position, make_approach_rotation_for_direction(vec(0.0, 0.0, -1.0))


# 배치 지점 위 경유점 후보 높이. 높은 쪽부터 시도한다.
CONVEYOR_TRANSIT_HEIGHTS_M = (0.30, 0.20, 0.15, 0.10, 0.05, 0.0)


def conveyor_transit_pose(runtime, place_position, place_rotation):
    """컨베이어 배치 직전 경유점 중 팔이 실제로 닿는 가장 높은 지점.

    배치 지점 위 0.30 m 를 고정으로 쓰면 안 된다. 컨베이어는 나무 반대편에
    있어서 배치 지점 자체가 이미 팔 길이의 거의 끝이고, 거기서 0.30 m 를 더
    올리면 반경을 넘는다. 실측에서 배치 지점은 base 로부터 1.529 m 로 IK 해가
    있었지만 0.30 m 위는 1.609 m 로 해가 없었고, TRANSPORT 가 timeout 났다.
    """
    place_position = np.asarray(place_position, dtype=float)
    for height in CONVEYOR_TRANSIT_HEIGHTS_M:
        candidate = place_position + vec(0.0, 0.0, height)
        if runtime.solve_ik(candidate, place_rotation) is not None:
            return candidate, height
    raise ApproachUnreachableError(
        f"컨베이어 배치 경유점에 도달할 수 없습니다: {np.round(place_position, 3)}"
    )


# ══════════════════════════════════════════════════════════════
# 부트스트랩
# ══════════════════════════════════════════════════════════════
CONVEYOR_SPEED_MPS = float(os.environ.get("HARVEST_CONVEYOR_SPEED", "0.3"))


def configure_conveyor_transport(stage, speed_mps=None):
    """덮개 평면을 움직이는 벨트로 만든다. 반드시 World.reset() **이후** 호출.

    공 낙하 실측으로 확정한 조리법이다. 순서와 시점이 전부 결과를 갈랐다.

    * 저장된 ConveyorBeltGraph 는 tick/velocity 연결이 전부 끊겨 있어(연결
      수 0) 롤러 이송이 영영 일어나지 않는다.
    * 롤러 위 직접 배치는 80 mm 사과가 롤러 틈에 낀다.
    * surface velocity 를 reset **전에** authoring 하면 적용되지 않았고,
      reset **후** 같은 API 를 적용하면 실제로 끌린다 (4초 -0.36 m 실측).
    * 평면 상면(0.728)이 롤러 상면(0.759~0.769)보다 낮아, 평면을 z=0.72
      (상면 0.77)로 올려 사과가 전 구간 평면을 타게 한다. 이동도 reset
      후 kinematic body 이동이라 문제없다.

    이송 방향은 배치 지점 -> 탑뷰 카메라(y=-0.03) 방향인 월드 -Y.
    """
    from pxr import Gf, UsdShade

    speed = float(CONVEYOR_SPEED_MPS if speed_mps is None else speed_mps)
    surface = stage.GetPrimAtPath(RUNTIME_CONVEYOR_COLLIDER_PATH)
    if not surface.IsValid():
        print(f"   [WARN] 덮개 평면이 없습니다: {RUNTIME_CONVEYOR_COLLIDER_PATH}")
        return False
    bound = 0
    with session_edit(stage):
        for op in UsdGeom.Xformable(surface).GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                t = op.Get()
                op.Set(Gf.Vec3d(t[0], t[1], 0.72))
                break
        UsdPhysics.CollisionAPI.Apply(surface).CreateCollisionEnabledAttr(True).Set(True)
        UsdPhysics.RigidBodyAPI.Apply(surface).CreateKinematicEnabledAttr(True).Set(True)
        sva = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(surface)
        sva.CreateSurfaceVelocityEnabledAttr(True).Set(True)
        sva.CreateSurfaceVelocityLocalSpaceAttr(False).Set(False)
        vec3 = Gf.Vec3f(0.0, -speed, 0.0)
        sva.CreateSurfaceVelocityAttr(vec3).Set(vec3)

        material = UsdShade.Material.Define(stage, "/World/RuntimeApplePhysicsMaterial")
        pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        pm.CreateStaticFrictionAttr(0.9).Set(0.9)
        pm.CreateDynamicFrictionAttr(0.8).Set(0.8)
        pm.CreateRestitutionAttr(0.05).Set(0.05)
        binding_targets = [surface]
        all_prims = tuple(stage.TraverseAll())
        for rigid in all_prims:
            if "apple" not in str(rigid.GetPath()).lower():
                continue
            if not rigid.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            for collider in all_prims:
                if not collider.GetPath().HasPrefix(rigid.GetPath()):
                    continue
                if not collider.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                binding_targets.append(collider)
        for prim in binding_targets:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            bound += 1
    print(
        f"   컨베이어 이송 설정: 평면 상면 0.77, surface velocity (0, -{speed:.2f}, 0), "
        f"마찰 재질 {bound}개 바인딩"
    )
    return True


def bootstrap(robot_ids=None, world=None, pre_reset_hook=None):
    """stage 를 열고 지정한 로봇들의 런타임을 준비한다.

    ``robot_ids`` 에 두 개를 주면 하나의 World 에서 두 로봇이 동시에
    움직인다. 구동하지 않는 로봇만 저장된 자세로 고정한다.
    """
    ids = tuple(robot_ids) if robot_ids else (ROBOT_PROFILE.robot_id,)
    for robot_id in ids:
        if robot_id not in ROBOT_RUNTIME_PROFILES:
            raise HarvestError(f"알 수 없는 robot_id 입니다: {robot_id}")

    stage = open_project_stage()
    print(f"   Stage : {STAGE_PATH.name}")
    print(f"   구동 로봇: {', '.join(ids)}")

    if world is None:
        world = World(stage_units_in_meters=1.0)

    runtimes = {}
    for robot_id in ids:
        runtimes[robot_id] = RobotRuntime(
            world, stage, ROBOT_RUNTIME_PROFILES[robot_id]
        )
    hold_idle_robots(stage, ids)

    # PhysX 는 초기화 시점의 스키마만 반영하므로, 물리 형상을 바꾸는 훅은
    # 반드시 reset 전에 실행한다.
    if pre_reset_hook is not None:
        pre_reset_hook(stage)

    # PhysX 파손 이벤트 구독은 물리가 돌기 전에 걸어 둔다.
    JOINT_BREAK_RECORDER.start()

    world.reset()
    # 이송 설정은 reset 후에만 물리에 반영된다 (공 낙하 실측).
    configure_conveyor_transport(stage)
    for runtime in runtimes.values():
        runtime.after_world_reset()
    for _ in range(60):
        world.step(render=True)

    return {"world": world, "stage": stage, "runtimes": runtimes}


def harvest_one_apple(runtime, assembly=None):
    """사과 하나를 수확해 컨베이어에 놓는다."""
    fsm = AppleHarvestFSM(runtime)
    try:
        if assembly is None:
            if not runtime.assemblies:
                raise HarvestError("수확할 사과 assembly 를 찾지 못했습니다.")
            assembly = runtime.assemblies[0]
        center = runtime.apple_center(assembly["apple"])
        fsm.select_apple(center)
        print(f"\n   [{runtime.robot_id}] 목표 사과 중심 {np.round(center, 3)}")

        fsm.approach(center)
        print("   APPROACH 완료")
        fsm.grasp()
        print("   GRASP 완료")
        fsm.twist()
        print("   TWIST 완료")
        fsm.linear_pull()
        print("   PULL 및 stem 분리 완료")

        place_position, place_rotation = conveyor_place_pose(runtime.stage)
        transit, height = conveyor_transit_pose(runtime, place_position, place_rotation)
        fsm.transport(transit, place_rotation)
        print(f"   TRANSPORT 완료 (경유 높이 {height * 100:.0f} cm)")
        fsm.place(place_position, place_rotation)
        print("   PLACE 완료")
        fsm.release()
        print("   RELEASE 완료")
        fsm.retract()
        print("   RETRACT 완료")
        return True
    finally:
        fsm.close()


def main():
    print("\n══════ M0617 사과 수확 ══════")
    context = None
    try:
        context = bootstrap()
        runtime = context["runtimes"][ROBOT_PROFILE.robot_id]
        harvest_one_apple(runtime)
        print("\n   수확 성공\n")
        steps = 0
        while simulation_app.is_running():
            context["world"].step(render=True)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
    except HarvestError as error:
        print(f"\n   [FAIL] {error.error_code}: {error}\n", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        if context is not None:
            for runtime in context["runtimes"].values():
                runtime.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
