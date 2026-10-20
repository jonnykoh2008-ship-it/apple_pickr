"""GPU PC 1용 통합 Isaac Sim 수확 서버 (로봇 2대, 월드 1개).

카메라 발행과 M0617 제어는 반드시 같은 Isaac Sim World에서 실행해야 한다.
컨베이어도 한 대뿐이므로 두 로봇은 같은 세계를 공유해야 한다. 그래서 이
프로세스 하나가 Isaac Sim 하나를 띄우고 그 안에서 로봇 두 대를 함께 굴린다.
로봇마다 프로세스를 띄우면 Isaac Sim이 두 개 뜨고 월드도 둘로 갈려서,
컨베이어가 각자 것이 되고 ``/clock``을 둘이 발행해 서로를 덮어쓴다.

이 파일은 ``apple_pick.py``의 물리·IK 함수를 재사용하고,
``/<robot_id>/harvest/robot_motion`` Action Goal 단위로 동작을 나눈다.

한 프로세스가 담당하는 것:
    * ``/clock`` (전역 1개)
    * 로봇별 base D455 RGB/depth/CameraInfo
    * 로봇별 TF와 ``/<robot_id>/joint_states``
    * ``/simulation/state`` 세대 (전역)
    * ``/planning_scene`` snapshot과 snapshot 서비스 (전역)
    * 로봇별 ``/<robot_id>/harvest/robot_motion`` Action 서버
    * 로봇별 ``/<robot_id>/harvest/motion_status``

실행 전 ``APPLEPROJ_INTERFACES_PREFIX``에는 Isaac Python 3.11로 빌드한
appleproj_interfaces의 install prefix를 지정해야 한다.

실행:
    PYTHONUNBUFFERED=1 ROS_DOMAIN_ID=103 ~/isaacsim/python.sh vision_apple_pick.py
    PYTHONUNBUFFERED=1 ROS_DOMAIN_ID=103 ~/isaacsim/python.sh vision_apple_pick.py \\
        --robots robot_01
"""

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ISAAC_SIM_ROOT = Path("/home/rokey/isaacsim")
ROS2_BRIDGE_ROOT = ISAAC_SIM_ROOT / "exts/isaacsim.ros2.bridge/jazzy"
ROS2_BRIDGE_LIB_DIR = ROS2_BRIDGE_ROOT / "lib"
# Isaac Sim은 ROS 2 Bridge 안에 Python 3.11용 rclpy를 번들로 넣어 둔다.
# 시스템 ROS 2의 rclpy는 3.12 전용이라 Isaac 안에서 import되지 않는다.
ROS2_BRIDGE_RCLPY_DIR = ROS2_BRIDGE_ROOT / "rclpy"
ENV_REEXEC_GUARD = "VISION_APPLE_PICK_ENV_READY"

os.environ.setdefault("ROS_DOMAIN_ID", "102")


def prepare_isaac_ros_environment():
    """Isaac Python 3.11용 rclpy와 custom 메시지를 쓸 환경을 만든다.

    ``LD_LIBRARY_PATH``
        ROS 2 Bridge의 Jazzy 라이브러리와, custom 메시지의 C 타입서포트
        ``.so`` 가 들어 있는 install prefix의 ``lib`` 이다. 둘 다 dlopen 이
        찾는 경로라서 프로세스가 시작한 뒤에 바꾸면 늦다. 빠진 경로가
        있으면 환경을 채워 현재 Python을 한 번만 재실행한다. 이게 없으면
        publisher를 만드는 순간
        ``libappleproj_interfaces__rosidl_typesupport_fastrtps_c.so``
        를 못 찾아 ``Type support not from this implementation`` 으로 죽는다.

    ``sys.path``
        rclpy는 Isaac이 번들한 3.11 빌드를, custom 메시지는
        ``build_interfaces_for_isaac.sh`` 가 3.11로 다시 컴파일한 것을 쓴다.
    """
    if not ROS2_BRIDGE_LIB_DIR.is_dir():
        raise RuntimeError(
            f"Isaac Sim ROS 2 Jazzy 라이브러리 폴더가 없습니다: {ROS2_BRIDGE_LIB_DIR}"
        )
    if not ROS2_BRIDGE_RCLPY_DIR.is_dir():
        raise RuntimeError(f"Isaac 번들 rclpy 폴더가 없습니다: {ROS2_BRIDGE_RCLPY_DIR}")

    text = os.environ.get("APPLEPROJ_INTERFACES_PREFIX", "").strip()
    if not text:
        raise RuntimeError(
            "APPLEPROJ_INTERFACES_PREFIX가 비어 있습니다. "
            "./build_interfaces_for_isaac.sh 를 먼저 실행하고 "
            "export APPLEPROJ_INTERFACES_PREFIX=$PWD/install_isaac311/appleproj_interfaces "
            "를 설정하세요."
        )
    prefix = Path(text)
    if not prefix.is_dir():
        raise RuntimeError(f"APPLEPROJ_INTERFACES_PREFIX 경로가 없습니다: {prefix}")

    required = [str(ROS2_BRIDGE_LIB_DIR), str(prefix / "lib")]
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [path for path in required if path not in current]
    if missing:
        if os.environ.get(ENV_REEXEC_GUARD) == "1":
            raise RuntimeError(
                f"LD_LIBRARY_PATH 보정 후에도 경로가 반영되지 않았습니다: {missing}"
            )
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = ":".join([*missing, *current])
        environment[ENV_REEXEC_GUARD] = "1"
        os.execve(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )

    # custom 메시지를 rclpy 번들보다 앞에 둔다. 같은 패키지 이름이 양쪽에
    # 있으면 먼저 찾은 쪽이 이긴다.
    site_packages = [
        str(candidate) for candidate in prefix.glob("lib/python3.*/site-packages")
    ]
    sys.path[:0] = [*site_packages, str(ROS2_BRIDGE_RCLPY_DIR)]
    return prefix


prepare_isaac_ros_environment()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# apple_pick가 SimulationApp을 만든다. 다른 Isaac 모듈보다 먼저 import해야 한다.
import apple_pick as harvest  # noqa: E402

import queue  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402

import omni.graph.core as og  # noqa: E402
import usdrt.Sdf  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import rclpy  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from appleproj_interfaces.action import RobotMotion  # noqa: E402
from appleproj_interfaces.msg import (  # noqa: E402
    MotionStatus,
    ObstacleProxy,
    PlanningScene,
    SimulationState,
)
from appleproj_interfaces.srv import GetPlanningScene  # noqa: E402
from geometry_msgs.msg import Pose, PoseStamped  # noqa: E402

from harvest_namespace import HarvestNames  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 실행 옵션
# ══════════════════════════════════════════════════════════════
def parse_robot_ids():
    """--robots 로 구동할 로봇을 고른다. 기본은 두 대 모두."""
    if "--robots" not in sys.argv:
        return harvest.ROBOT_IDS
    index = sys.argv.index("--robots")
    selected = []
    for value in sys.argv[index + 1 :]:
        if value.startswith("--"):
            break
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if item not in harvest.ROBOT_RUNTIME_PROFILES:
                raise SystemExit(f"알 수 없는 robot id 입니다: {item}")
            selected.append(item)
    if not selected:
        raise SystemExit("--robots 에 로봇을 하나 이상 지정하세요.")
    return tuple(dict.fromkeys(selected))


ROBOT_IDS = parse_robot_ids()

CAMERA_WIDTH = int(os.environ.get("HARVEST_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("HARVEST_CAMERA_HEIGHT", "720"))
CAMERA_FRAME_SKIP = int(os.environ.get("HARVEST_CAMERA_FRAME_SKIP", "1"))

CLOCK_GRAPH_PATH = "/HarvestClockRosGraph"

# 컨베이어 탑뷰 카메라. 이름과 frame 은 GPU PC 2 의
# conveyor_camera_adapter_node 가 구독하는 계약
# (docs/architecture/ros2_interfaces.md) 그대로다.
CONVEYOR_CAMERA_ROOT = "/World/conv_rsd455"
CONVEYOR_CAMERA_NS = "/conveyor_camera"
CONVEYOR_CAMERA_FRAME = "quality_camera_top_optical_frame"
CONVEYOR_CAMERA_GRAPH_PATH = "/ConveyorCameraRosGraph"

# 컨베이어 벨트 명령 속도. 저장된 USD 의 ConveyorBeltGraph 에는 Velocity
# 변수값이 authored 되어 있지 않아, 실행 시 여기서 넣지 않으면 벨트가 0 으로
# 서 있고 사과가 배치 지점에 머물러 검사 ROI 에 영영 들어가지 않는다.
# 예전에는 conveyor_camera_publish.py 가 이 값을 세팅했는데 그 프로세스를
# 이 서버로 통합하면서 속도 설정도 함께 가져왔다.
CONVEYOR_BELT_GRAPH_PATH = "/World/ConveyorTrack_01/ConveyorBeltGraph"
CONVEYOR_SPEED_MPS = float(os.environ.get("HARVEST_CONVEYOR_SPEED", "0.3"))

# 오류 코드는 docs/architecture/ros2_interfaces.md 의 300번대 체계를 쓴다.
ERROR_CODES = {
    "GOAL_REJECTED": "306:GOAL_REJECTED",
    "CANCELLED": "307:CANCELLED",
    "SIMULATION_RESET": "308:SIMULATION_RESET",
    "INTERNAL_ERROR": "312:INTERNAL_ERROR",
}

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


# ══════════════════════════════════════════════════════════════
# ROS 2 Bridge 그래프
# ══════════════════════════════════════════════════════════════
def quaternion_xyzw_from_matrix(matrix):
    """회전행렬을 ROS 순서 [x, y, z, w] 쿼터니언으로 바꾼다."""
    return harvest.quat_wxyz_to_xyzw(
        harvest.rot_matrix_to_quat(np.asarray(matrix, dtype=float))
    )


# USD 카메라는 로컬 -Z 를 보고 +Y 가 위다. ROS 광학 프레임은 +Z 가 전방이고
# +Y 가 아래다. 두 규약은 X 축 180도 회전만큼 다르다.
#
# 이 변환 없이 카메라 prim 의 자세를 그대로 TF 로 내보내면, 구독자가
# depth 로 역투영한 점(광학 규약)을 엉뚱한 축으로 world 에 옮긴다. 실측에서
# 실제 사과가 (0.87, 0.08, 1.01) 인데 (2.71, -0.41, 2.41) 로 나왔다.
USD_CAMERA_TO_ROS_OPTICAL = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
)


def camera_optical_frame_pose(stage, camera_path):
    """카메라 광학 프레임의 world (position, rotation)."""
    position, rotation = harvest.get_prim_world_pose(stage, camera_path)
    return position, rotation @ USD_CAMERA_TO_ROS_OPTICAL


def build_clock_graph(stage):
    """``/clock`` 을 전역으로 한 번만 발행한다.

    로봇마다 발행하면 두 publisher 가 같은 토픽에 써서 구독자가 받는
    시각이 둘 사이를 오간다.
    """
    keys = og.Controller.Keys
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": CLOCK_GRAPH_PATH,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishClock.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [("PublishClock.inputs:topicName", "/clock")],
            },
        )
    print("   /clock 그래프 생성 (전역 1개)")


def build_camera_graph(stage, profile, names):
    """한 로봇의 base D455 RGB/depth/CameraInfo와 카메라 TF를 발행한다."""
    camera_path = profile.camera_prim_path
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise harvest.HarvestError(
            f"D455 payload 안의 Color Camera 가 로드되지 않았습니다: {camera_path}. "
            "에셋 서버 연결 또는 로컬 캐시를 확인하세요."
        )
    translation, rotation = camera_optical_frame_pose(stage, camera_path)
    quaternion = quaternion_xyzw_from_matrix(rotation)
    graph_path = f"/BaseCameraRosGraph_{profile.robot_id}"

    keys = og.Controller.Keys
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    (
                        "CreateRenderProduct",
                        "isaacsim.core.nodes.IsaacCreateRenderProduct",
                    ),
                    ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("PublishDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    (
                        "PublishCameraInfo",
                        "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
                    ),
                    (
                        "PublishCameraTf",
                        "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                    ),
                ],
                keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "CreateRenderProduct.inputs:execIn",
                    ),
                    ("CreateRenderProduct.outputs:execOut", "PublishRgb.inputs:execIn"),
                    (
                        "CreateRenderProduct.outputs:execOut",
                        "PublishDepth.inputs:execIn",
                    ),
                    (
                        "CreateRenderProduct.outputs:execOut",
                        "PublishCameraInfo.inputs:execIn",
                    ),
                    (
                        "CreateRenderProduct.outputs:renderProductPath",
                        "PublishRgb.inputs:renderProductPath",
                    ),
                    (
                        "CreateRenderProduct.outputs:renderProductPath",
                        "PublishDepth.inputs:renderProductPath",
                    ),
                    (
                        "CreateRenderProduct.outputs:renderProductPath",
                        "PublishCameraInfo.inputs:renderProductPath",
                    ),
                    ("OnPlaybackTick.outputs:tick", "PublishCameraTf.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishCameraTf.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    (
                        "CreateRenderProduct.inputs:cameraPrim",
                        [usdrt.Sdf.Path(camera_path)],
                    ),
                    ("CreateRenderProduct.inputs:width", CAMERA_WIDTH),
                    ("CreateRenderProduct.inputs:height", CAMERA_HEIGHT),
                    ("PublishRgb.inputs:frameId", names.camera_frame),
                    ("PublishRgb.inputs:topicName", names.rgb_topic),
                    ("PublishRgb.inputs:type", "rgb"),
                    ("PublishRgb.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                    ("PublishDepth.inputs:frameId", names.camera_frame),
                    ("PublishDepth.inputs:topicName", names.depth_topic),
                    ("PublishDepth.inputs:type", "depth"),
                    ("PublishDepth.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                    ("PublishCameraInfo.inputs:frameId", names.camera_frame),
                    ("PublishCameraInfo.inputs:topicName", names.camera_info_topic),
                    ("PublishCameraInfo.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                    ("PublishCameraTf.inputs:topicName", "/tf_static"),
                    ("PublishCameraTf.inputs:parentFrameId", "world"),
                    ("PublishCameraTf.inputs:childFrameId", names.camera_frame),
                    ("PublishCameraTf.inputs:staticPublisher", True),
                    ("PublishCameraTf.inputs:translation", translation.tolist()),
                    ("PublishCameraTf.inputs:rotation", quaternion.tolist()),
                ],
            },
        )
    print(f"   [{profile.robot_id}] 카메라 그래프 {graph_path}")
    print(f"       RGB   {names.rgb_topic}")
    print(f"       Depth {names.depth_topic}")
    print(f"       Info  {names.camera_info_topic}")
    print(f"       frame {names.camera_frame}")


def build_conveyor_camera_graph(stage):
    """컨베이어 탑뷰 D455 를 품질검사용 토픽으로 발행한다.

    GPU PC 2 의 conveyor_camera_adapter_node 가 구독하는 이름과 frame 을
    그대로 쓴다. 예전에는 conveyor_camera_publish.py 가 별도 Isaac Sim
    프로세스로 이걸 발행했는데, 그러면 Isaac 이 두 개 뜨고 월드가 갈려서
    수확한 사과와 검사받는 사과가 서로 다른 세계에 있게 된다. 카메라는
    로봇과 같은 월드에서 발행해야 한다.

    현재 stage 에는 탑뷰 conv_rsd455 한 대만 있다. 왼쪽·오른쪽
    (conv_rsd455_01/_02) 프림이 추가되면 이 함수를 그 카메라에도 부르면
    된다.
    """
    camera_path = f"{CONVEYOR_CAMERA_ROOT}/RSD455/Camera_OmniVision_OV9782_Color"
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        print(
            f"   [WARN] 컨베이어 탑뷰 카메라가 없어 품질 스트림을 건너뜁니다: "
            f"{camera_path}"
        )
        return False
    translation, rotation = camera_optical_frame_pose(stage, camera_path)
    quaternion = quaternion_xyzw_from_matrix(rotation)

    keys = og.Controller.Keys
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": CONVEYOR_CAMERA_GRAPH_PATH,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    (
                        "CreateRenderProduct",
                        "isaacsim.core.nodes.IsaacCreateRenderProduct",
                    ),
                    ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("PublishDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    (
                        "PublishCameraInfo",
                        "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
                    ),
                    (
                        "PublishCameraTf",
                        "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                    ),
                ],
                keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "CreateRenderProduct.inputs:execIn",
                    ),
                    ("CreateRenderProduct.outputs:execOut", "PublishRgb.inputs:execIn"),
                    (
                        "CreateRenderProduct.outputs:execOut",
                        "PublishDepth.inputs:execIn",
                    ),
                    (
                        "CreateRenderProduct.outputs:execOut",
                        "PublishCameraInfo.inputs:execIn",
                    ),
                    (
                        "CreateRenderProduct.outputs:renderProductPath",
                        "PublishRgb.inputs:renderProductPath",
                    ),
                    (
                        "CreateRenderProduct.outputs:renderProductPath",
                        "PublishDepth.inputs:renderProductPath",
                    ),
                    (
                        "CreateRenderProduct.outputs:renderProductPath",
                        "PublishCameraInfo.inputs:renderProductPath",
                    ),
                    ("OnPlaybackTick.outputs:tick", "PublishCameraTf.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishCameraTf.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    (
                        "CreateRenderProduct.inputs:cameraPrim",
                        [usdrt.Sdf.Path(camera_path)],
                    ),
                    ("CreateRenderProduct.inputs:width", CAMERA_WIDTH),
                    ("CreateRenderProduct.inputs:height", CAMERA_HEIGHT),
                    ("PublishRgb.inputs:frameId", CONVEYOR_CAMERA_FRAME),
                    ("PublishRgb.inputs:topicName", f"{CONVEYOR_CAMERA_NS}/color/image_raw"),
                    ("PublishRgb.inputs:type", "rgb"),
                    ("PublishRgb.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                    ("PublishDepth.inputs:frameId", CONVEYOR_CAMERA_FRAME),
                    ("PublishDepth.inputs:topicName", f"{CONVEYOR_CAMERA_NS}/depth/image_raw"),
                    ("PublishDepth.inputs:type", "depth"),
                    ("PublishDepth.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                    ("PublishCameraInfo.inputs:frameId", CONVEYOR_CAMERA_FRAME),
                    (
                        "PublishCameraInfo.inputs:topicName",
                        f"{CONVEYOR_CAMERA_NS}/camera_info",
                    ),
                    ("PublishCameraInfo.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                    ("PublishCameraTf.inputs:topicName", "/tf_static"),
                    ("PublishCameraTf.inputs:parentFrameId", "world"),
                    ("PublishCameraTf.inputs:childFrameId", CONVEYOR_CAMERA_FRAME),
                    ("PublishCameraTf.inputs:staticPublisher", True),
                    ("PublishCameraTf.inputs:translation", translation.tolist()),
                    ("PublishCameraTf.inputs:rotation", quaternion.tolist()),
                ],
            },
        )
    print("   컨베이어 탑뷰 카메라 그래프 생성")
    print(f"       RGB   {CONVEYOR_CAMERA_NS}/color/image_raw")
    print(f"       Depth {CONVEYOR_CAMERA_NS}/depth/image_raw")
    print(f"       Info  {CONVEYOR_CAMERA_NS}/camera_info")
    print(f"       frame {CONVEYOR_CAMERA_FRAME}")
    return True


def set_conveyor_speed(stage):
    """벨트 이송이 실제로 사과를 움직이게 만든다.

    속도 변수만 넣는 것으로는 부족했다. 두 가지가 더 필요하다.

    1. ``RuntimeConveyorBeltSurface`` 는 벨트 위를 덮은 정적 평면이라,
       사과가 움직이는 벨트가 아니라 안 움직이는 이 평면 위에 얹힌다.
       배치 목표 계산에는 계속 쓰되 물리 collision 은 끈다.
    2. 사과 collider 에 마찰 재질을 묶는다. 마찰이 없으면 벨트가 지나가도
       사과가 미끄러져 제자리에 남는다.

    전부 Session Layer 에만 적용한다. 예전에는 conveyor_camera_publish.py
    가 이걸 자기 월드에서 했는데, 그 프로세스를 이 서버로 통합했다.
    """
    graph = stage.GetPrimAtPath(CONVEYOR_BELT_GRAPH_PATH)
    if not graph.IsValid():
        print(f"   [WARN] 컨베이어 그래프가 없습니다: {CONVEYOR_BELT_GRAPH_PATH}")
        return False
    velocity = graph.GetAttribute("graph:variable:Velocity")
    if not velocity.IsValid():
        print("   [WARN] ConveyorBeltGraph 에 Velocity 변수가 없습니다.")
        return False

    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    bound = 0
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        velocity.Set(float(CONVEYOR_SPEED_MPS))

        # 덮개 평면 물리는 configure_conveyor_transport 가 reset 전에 만든다.

        material = UsdShade.Material.Define(stage, "/World/RuntimeApplePhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(0.9).Set(0.9)
        physics_material.CreateDynamicFrictionAttr(0.8).Set(0.8)
        physics_material.CreateRestitutionAttr(0.05).Set(0.05)

        all_prims = tuple(stage.TraverseAll())
        for rigid in all_prims:
            if "apple" not in str(rigid.GetPath()).lower():
                continue
            if not rigid.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if bool(rigid.GetAttribute("physics:kinematicEnabled").Get()):
                continue
            for collider in all_prims:
                if not collider.GetPath().HasPrefix(rigid.GetPath()):
                    continue
                if not collider.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bound += 1
    print(
        f"   컨베이어 이송 설정: 덮개 surface velocity (0, -{CONVEYOR_SPEED_MPS:.2f}, 0) m/s, "
        f"사과 마찰 재질 {bound}개 바인딩"
    )
    return True


def build_plate_conveyor_graph(stage):
    """덮개 평면을 IsaacConveyor 노드로 구동한다.

    PhysxSurfaceVelocityAPI 를 USD 로 authoring 하는 방식은 이 버전에서
    적용되지 않았다(공 낙하 실측: 드리프트만 있고 이송 없음). 스테이지의
    롤러 그래프가 쓰는 것과 같은 IsaacConveyor 노드는 런타임 physx API 로
    매 스텝 surface velocity 를 넣으므로 확실하다.
    """
    keys = og.Controller.Keys
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": "/ConveyorPlateGraph",
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Conveyor", "isaacsim.asset.gen.conveyor.IsaacConveyor"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "Conveyor.inputs:onStep"),
                    ("OnPlaybackTick.outputs:deltaSeconds", "Conveyor.inputs:delta"),
                ],
                keys.SET_VALUES: [
                    (
                        "Conveyor.inputs:conveyorPrim",
                        [usdrt.Sdf.Path(harvest.RUNTIME_CONVEYOR_COLLIDER_PATH)],
                    ),
                    ("Conveyor.inputs:enabled", True),
                    ("Conveyor.inputs:velocity", float(CONVEYOR_SPEED_MPS)),
                    # direction 은 대상 prim 로컬 기준. 덮개는 회전이 없어
                    # 로컬 -Y == 월드 -Y (배치 지점 -> 탑뷰 카메라 방향).
                    ("Conveyor.inputs:direction", [0.0, -1.0, 0.0]),
                ],
            },
        )
    print(f"   덮개 컨베이어 그래프 생성: {CONVEYOR_SPEED_MPS:.2f} m/s, -Y")


def build_robot_state_graph(stage, profile, names):
    """한 로봇의 링크 TF와 joint_states 를 발행한다.

    robot_state_publisher 는 쓰지 않는다. TF 의 권위자는 Isaac Sim 이고
    같은 TF 를 두 노드가 중복 발행하지 않는다.
    """
    graph_path = f"/RobotStateRosGraph_{profile.robot_id}"
    keys = og.Controller.Keys
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishTf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishTf.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTf.inputs:timeStamp"),
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    ("PublishTf.inputs:topicName", "/tf"),
                    (
                        "PublishTf.inputs:targetPrims",
                        [usdrt.Sdf.Path(profile.robot_prim_path)],
                    ),
                    ("PublishJointState.inputs:topicName", names.joint_states_topic),
                    (
                        # ArticulationRootAPI 가 붙어 있는 prim 을 정확히
                        # 가리켜야 한다. 부모 Xform 을 주면 이 노드는
                        # "is not an articulation" 으로 조용히 실패하고
                        # joint_states 토픽 자체가 생기지 않는다.
                        "PublishJointState.inputs:targetPrim",
                        [usdrt.Sdf.Path(profile.articulation_root_joint_path)],
                    ),
                ],
            },
        )
    print(f"   [{profile.robot_id}] 로봇 상태 그래프 {graph_path}")
    print(f"       joint_states {names.joint_states_topic}")


# ══════════════════════════════════════════════════════════════
# 시뮬레이션 스레드에 넘길 작업
# ══════════════════════════════════════════════════════════════
@dataclass
class MotionJob:
    """Action 콜백이 만들고 시뮬레이션 스레드가 실행하는 한 단계.

    Isaac Sim API 는 스레드 안전하지 않다. Action 콜백에서 직접 world 를
    step 하면 렌더 스레드와 충돌하므로, 작업만 넘기고 결과를 기다린다.
    큐가 하나라서 두 로봇의 단계가 한 번에 하나씩 순서대로 실행된다.
    """

    robot_id: str
    motion_type: int
    target_position: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)
    success: bool = False
    error_code: str = ""
    message: str = ""
    progress: float = 0.0


class HarvestWorld:
    """한 Isaac Sim World 와 그 안의 로봇 런타임 전부."""

    def __init__(self, robot_ids):
        # 이송 설정은 apple_pick.bootstrap 이 reset 직후 직접 수행한다.
        context = harvest.bootstrap(robot_ids)
        self.world = context["world"]
        self.stage = context["stage"]
        self.runtimes = context["runtimes"]
        self.fsms = {
            robot_id: harvest.AppleHarvestFSM(runtime)
            for robot_id, runtime in self.runtimes.items()
        }
        self.reset_id = 0
        self.scene_version = 0

    def close(self):
        for fsm in self.fsms.values():
            fsm.close()
        for runtime in self.runtimes.values():
            runtime.close()

    def all_proxies(self):
        """두 나무의 proxy 를 합친 전역 snapshot."""
        merged = []
        for runtime in self.runtimes.values():
            merged.extend(runtime.proxies)
        return tuple(merged)

    def refresh_scene_if_moved(self):
        changed = False
        for runtime in self.runtimes.values():
            if runtime.refresh_scene_if_moved():
                changed = True
        if changed:
            self.scene_version += 1
        return changed

    def execute(self, job):
        """한 Goal 단계를 실행한다. 시뮬레이션 스레드에서만 호출한다."""
        runtime = self.runtimes[job.robot_id]
        fsm = self.fsms[job.robot_id]
        motion_type = job.motion_type

        if motion_type == RobotMotion.Goal.APPROACH:
            _assembly, center = fsm.select_apple(job.target_position)
            fsm.approach(center)
        elif motion_type == RobotMotion.Goal.GRASP:
            fsm.grasp()
        elif motion_type == RobotMotion.Goal.TWIST:
            fsm.twist()
        elif motion_type == RobotMotion.Goal.PULL:
            fsm.linear_pull()
        elif motion_type == RobotMotion.Goal.TRANSPORT:
            position, rotation = harvest.conveyor_place_pose(self.stage)
            transit, height = harvest.conveyor_transit_pose(runtime, position, rotation)
            print(f"   [{job.robot_id}] 컨베이어 경유 높이 {height * 100:.0f} cm")
            fsm.transport(transit, rotation)
        elif motion_type == RobotMotion.Goal.PLACE:
            position, rotation = harvest.conveyor_place_pose(self.stage)
            fsm.place(position, rotation)
        elif motion_type == RobotMotion.Goal.RELEASE:
            fsm.release()
        elif motion_type == RobotMotion.Goal.RETRACT:
            fsm.retract()
        else:
            raise harvest.HarvestError(
                f"지원하지 않는 motion_type 입니다: {motion_type}"
            )
        return True


# ══════════════════════════════════════════════════════════════
# 로봇별 Action 서버
# ══════════════════════════════════════════════════════════════
class RobotMotionServer:
    """한 로봇의 Action 서버와 상태 발행."""

    def __init__(self, node, robot_id, job_queue, state_provider):
        self.node = node
        self.robot_id = robot_id
        self.names = HarvestNames(robot_id)
        self.job_queue = job_queue
        self.state_provider = state_provider
        self.busy = threading.Lock()
        self.active_job = None

        callback_group = ReentrantCallbackGroup()
        self.motion_status_publisher = node.create_publisher(
            MotionStatus, self.names.motion_status_topic, RELIABLE_QOS
        )
        self.action_server = ActionServer(
            node,
            RobotMotion,
            self.names.robot_motion_action,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=callback_group,
        )
        node.get_logger().info(
            f"Action 서버 준비: {self.names.robot_motion_action}"
        )

    def publish_status(self, state, success, progress, error_code="", text=""):
        message = MotionStatus()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.current_state = state
        message.success = bool(success)
        message.progress = float(progress)
        message.error_code = error_code
        message.message = text
        self.motion_status_publisher.publish(message)

    def goal_callback(self, goal_request):
        """세대와 busy 상태를 함께 검사해 Goal 을 승인한다."""
        snapshot = self.state_provider()
        if snapshot["state"] not in (SimulationState.READY, SimulationState.PLAYING):
            self.node.get_logger().warning(
                f"[{self.robot_id}] simulation 이 READY/PLAYING 이 아니라 거부"
            )
            return GoalResponse.REJECT
        if goal_request.reset_id != snapshot["reset_id"]:
            self.node.get_logger().warning(
                f"[{self.robot_id}] reset_id 불일치: "
                f"{goal_request.reset_id} != {snapshot['reset_id']}"
            )
            return GoalResponse.REJECT
        if goal_request.scene_version != snapshot["scene_version"]:
            self.node.get_logger().warning(
                f"[{self.robot_id}] scene_version 불일치: "
                f"{goal_request.scene_version} != {snapshot['scene_version']}"
            )
            return GoalResponse.REJECT
        if self.busy.locked():
            self.node.get_logger().warning(
                f"[{self.robot_id}] 실행 중에는 새 Goal 을 받지 않습니다."
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        if self.active_job is not None:
            self.active_job.cancel.set()
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        name = MOTION_NAMES.get(request.motion_type, "UNKNOWN")
        result = RobotMotion.Result()

        with self.busy:
            position = np.array(
                [
                    request.target_pose.pose.position.x,
                    request.target_pose.pose.position.y,
                    request.target_pose.pose.position.z,
                ],
                dtype=float,
            )
            job = MotionJob(
                robot_id=self.robot_id,
                motion_type=request.motion_type,
                target_position=position,
            )
            self.active_job = job
            self.publish_status(name, True, 0.0, "", f"{name} 시작")
            self.job_queue.put(job)

            while not job.done.wait(timeout=0.1):
                if goal_handle.is_cancel_requested:
                    job.cancel.set()
                feedback = RobotMotion.Feedback()
                feedback.current_state = name
                feedback.progress = float(job.progress)
                goal_handle.publish_feedback(feedback)

            self.active_job = None

        if job.cancel.is_set() and not job.success:
            goal_handle.canceled()
            result.success = False
            result.error_code = ERROR_CODES["CANCELLED"]
            result.message = f"{name} 취소됨"
        elif job.success:
            goal_handle.succeed()
            result.success = True
            result.error_code = ""
            result.message = f"{name} 완료"
        else:
            goal_handle.abort()
            result.success = False
            result.error_code = job.error_code or ERROR_CODES["INTERNAL_ERROR"]
            result.message = job.message

        self.publish_status(
            name,
            result.success,
            1.0 if result.success else job.progress,
            result.error_code,
            result.message,
        )
        return result


# ══════════════════════════════════════════════════════════════
# ROS 2 노드
# ══════════════════════════════════════════════════════════════
class HarvestServerNode(Node):
    """전역 상태 발행과 로봇별 Action 서버를 담는 노드."""

    def __init__(self, robot_ids, job_queue, state_provider):
        super().__init__("harvest_server")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.state_provider = state_provider
        self._last_scene = None

        self.simulation_state_publisher = self.create_publisher(
            SimulationState, HarvestNames.simulation_state_topic, LATCHED_QOS
        )
        self.planning_scene_publisher = self.create_publisher(
            PlanningScene, HarvestNames.planning_scene_topic, LATCHED_QOS
        )
        self.snapshot_service = self.create_service(
            GetPlanningScene,
            HarvestNames.planning_scene_service,
            self.on_snapshot_request,
            callback_group=ReentrantCallbackGroup(),
        )
        self.servers = {
            robot_id: RobotMotionServer(self, robot_id, job_queue, state_provider)
            for robot_id in robot_ids
        }
        self.create_timer(0.2, self.publish_simulation_state)

    def publish_simulation_state(self):
        snapshot = self.state_provider()
        message = SimulationState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.state = snapshot["state"]
        message.reset_id = snapshot["reset_id"]
        message.scene_version = snapshot["scene_version"]
        message.message = snapshot["message"]
        self.simulation_state_publisher.publish(message)

    def publish_planning_scene(self, snapshot):
        """정적 나무 proxy snapshot 을 전체 한 개로 발행한다.

        컨베이어와 마찬가지로 planning scene 은 하나의 세계를 기술하므로
        전역이다. 두 나무의 proxy 를 합쳐 보낸다.
        """
        message = PlanningScene()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.reset_id = snapshot["reset_id"]
        message.scene_version = snapshot["scene_version"]
        message.robot_base_pose = self._pose_stamped(
            snapshot["robot_base_position"], snapshot["robot_base_rotation"]
        )
        message.robot_tcp_pose = self._pose_stamped(
            snapshot["tcp_position"], snapshot["tcp_rotation"]
        )
        message.obstacles = [self._obstacle(spec) for spec in snapshot["proxies"]]
        self.planning_scene_publisher.publish(message)
        self._last_scene = message

    def _pose_stamped(self, position, rotation):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        quaternion = quaternion_xyzw_from_matrix(rotation)
        pose.pose.orientation.x = float(quaternion[0])
        pose.pose.orientation.y = float(quaternion[1])
        pose.pose.orientation.z = float(quaternion[2])
        pose.pose.orientation.w = float(quaternion[3])
        return pose

    @staticmethod
    def _obstacle(spec):
        obstacle = ObstacleProxy()
        obstacle.obstacle_id = spec.obstacle_id
        obstacle.shape = int(spec.shape)
        obstacle.obstacle_class = int(spec.obstacle_class)
        pose = Pose()
        pose.position.x = float(spec.position[0])
        pose.position.y = float(spec.position[1])
        pose.position.z = float(spec.position[2])
        pose.orientation.x = float(spec.orientation_xyzw[0])
        pose.orientation.y = float(spec.orientation_xyzw[1])
        pose.orientation.z = float(spec.orientation_xyzw[2])
        pose.orientation.w = float(spec.orientation_xyzw[3])
        obstacle.pose = pose
        obstacle.dimensions.x = float(spec.dimensions[0])
        obstacle.dimensions.y = float(spec.dimensions[1])
        obstacle.dimensions.z = float(spec.dimensions[2])
        obstacle.safety_margin = float(spec.safety_margin)
        return obstacle

    def on_snapshot_request(self, _request, response):
        if self._last_scene is None:
            response.success = False
            response.message = "아직 planning scene snapshot 을 만들지 않았습니다."
            return response
        response.success = True
        response.scene = self._last_scene
        response.message = ""
        return response


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
def error_code_for(error):
    """apple_pick 예외를 300번대 코드 문자열로 바꾼다."""
    if isinstance(error, harvest.HarvestError):
        return error.error_code
    return ERROR_CODES["INTERNAL_ERROR"]


def main():
    print("\n══════ GPU PC 1 Isaac 수확 서버 ══════")
    print(f"   구동 로봇 : {', '.join(ROBOT_IDS)}")
    print(f"   ROS_DOMAIN: {os.environ.get('ROS_DOMAIN_ID')}")

    enable_extension("isaacsim.ros2.bridge")
    # 컨베이어 OmniGraph 노드 타입이 이 확장에 있다. 켜지 않으면 저장된
    # ConveyorBeltGraph 가 로드되지 않아 벨트가 서 있고, 사과를 올려놓아도
    # 검사 구간으로 흘러가지 않는다.
    enable_extension("isaacsim.asset.gen.conveyor")
    harvest.simulation_app.update()

    world = None
    node = None
    executor = None
    job_queue = queue.Queue()

    try:
        world = HarvestWorld(ROBOT_IDS)

        build_clock_graph(world.stage)
        build_conveyor_camera_graph(world.stage)
        set_conveyor_speed(world.stage)
        for robot_id in ROBOT_IDS:
            profile = harvest.ROBOT_RUNTIME_PROFILES[robot_id]
            names = HarvestNames(robot_id)
            build_camera_graph(world.stage, profile, names)
            build_robot_state_graph(world.stage, profile, names)

        state = {"state": SimulationState.INITIALIZING, "message": "초기화 중"}
        first_runtime = world.runtimes[ROBOT_IDS[0]]

        def state_provider():
            base_position, base_rotation = first_runtime.base_pose()
            tcp_position, tcp_rotation = first_runtime.current_tcp_pose()
            return {
                "state": state["state"],
                "reset_id": world.reset_id,
                "scene_version": world.scene_version,
                "message": state["message"],
                "robot_base_position": base_position,
                "robot_base_rotation": base_rotation,
                "tcp_position": tcp_position,
                "tcp_rotation": tcp_rotation,
                "proxies": world.all_proxies(),
            }

        rclpy.init()
        node = HarvestServerNode(ROBOT_IDS, job_queue, state_provider)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()

        world.world.play()
        state["state"] = SimulationState.PLAYING
        state["message"] = "Timeline PLAYING"
        node.publish_planning_scene(state_provider())
        print("\n   준비 완료. Goal 을 기다립니다.\n")

        steps = 0
        while harvest.simulation_app.is_running():
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                job = None

            if job is not None:
                try:
                    job.progress = 0.1
                    world.execute(job)
                    job.progress = 1.0
                    job.success = True
                except harvest.HarvestError as error:
                    job.success = False
                    job.error_code = error_code_for(error)
                    job.message = str(error)
                    print(
                        f"   [FAIL] {job.robot_id} {job.error_code}: {error}",
                        file=sys.stderr,
                    )
                except Exception as error:  # noqa: BLE001
                    job.success = False
                    job.error_code = ERROR_CODES["INTERNAL_ERROR"]
                    job.message = str(error)
                    traceback.print_exc()
                finally:
                    job.done.set()
            else:
                world.world.step(render=True)
                steps += 1
                if steps % 120 == 0 and world.refresh_scene_if_moved():
                    print(f"   나무 이동 감지. scene_version -> {world.scene_version}")
                    node.publish_planning_scene(state_provider())

            if harvest.args.max_steps and steps >= harvest.args.max_steps:
                break
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if world is not None:
            world.close()
        harvest.simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
