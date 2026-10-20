"""Isaac Sim의 base_rsd455 영상을 ROS 2로 발행한다.

이 파일은 Isaac Sim에 포함된 Python 3.11로만 실행한다. 시스템 ROS 2의
``rclpy``를 직접 import하지 않고 ROS 2 Bridge OmniGraph 노드를 사용하므로,
Ubuntu 24.04 / ROS 2 Jazzy의 Python 3.12와 충돌하지 않는다.

발행 토픽 (<robot>은 --robot-id, 예: robot_01):
    /clock                                 rosgraph_msgs/msg/Clock
    /<robot>/base_camera/color/image_raw   sensor_msgs/msg/Image (rgb8)
    /<robot>/base_camera/depth/image_raw   sensor_msgs/msg/Image (32FC1, meter)
    /<robot>/base_camera/camera_info       sensor_msgs/msg/CameraInfo
    /tf_static                             world -> <robot>/base_camera

이름은 harvest_namespace 가 만든다. 로봇마다 다른 이름을 써야 두 대를 동시에
띄웠을 때 수신 측이 어느 로봇의 영상인지 구분할 수 있다.

실행 예시:
    ROS_DOMAIN_ID=102 /home/rokey/isaacsim/python.sh base_camera_publish.py
"""

import argparse
import os
import sys
from pathlib import Path

from harvest_namespace import HarvestNames


# 개별 비전 테스트의 기본 Domain이다. 실행 환경에서 명시한 값이 있으면
# 그 값을 우선하므로 이후 통합 단계에서는 외부에서 101로 바꿀 수 있다.
os.environ.setdefault("ROS_DOMAIN_ID", "102")


# ROS 2 Bridge의 Jazzy 라이브러리는 Isaac Sim 기본 검색 경로에 포함되지
# 않는 경우가 있다. Bridge가 로드된 뒤에는 LD_LIBRARY_PATH 변경이 늦으므로,
# SimulationApp을 import하기 전에 경로를 보정하고 현재 Python을 한 번 재실행한다.
ISAAC_SIM_ROOT = Path("/home/rokey/isaacsim")
ROS2_BRIDGE_LIB_DIR = (
    ISAAC_SIM_ROOT / "exts/isaacsim.ros2.bridge/jazzy/lib"
)
ENV_REEXEC_GUARD = "BASE_CAMERA_PUBLISH_ENV_READY"


def ensure_ros2_bridge_library_path():
    """ROS 2 Bridge 공유 라이브러리 경로를 실행 초기에 확정한다."""
    if not ROS2_BRIDGE_LIB_DIR.is_dir():
        raise RuntimeError(
            "Isaac Sim ROS 2 Jazzy 라이브러리 폴더가 없습니다: "
            f"{ROS2_BRIDGE_LIB_DIR}"
        )

    current_paths = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    ]
    bridge_path = str(ROS2_BRIDGE_LIB_DIR)
    if bridge_path in current_paths:
        return

    if os.environ.get(ENV_REEXEC_GUARD) == "1":
        raise RuntimeError(
            "LD_LIBRARY_PATH 자동 보정 후에도 ROS 2 Bridge 경로가 반영되지 "
            f"않았습니다: {bridge_path}"
        )

    updated_environment = os.environ.copy()
    updated_environment["LD_LIBRARY_PATH"] = ":".join(
        [bridge_path, *current_paths]
    )
    updated_environment[ENV_REEXEC_GUARD] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        updated_environment,
    )


ensure_ros2_bridge_library_path()

import numpy as np

from isaacsim import SimulationApp


def parse_arguments():
    """Isaac Sim을 시작하기 전에 실행 옵션을 읽는다."""
    parser = argparse.ArgumentParser(description="base_rsd455 ROS 2 publisher")
    parser.add_argument(
        "--robot-id",
        choices=("robot_01", "robot_02"),
        default="robot_01",
        help="USD base D455 선택 (기본값: robot_01)",
    )
    parser.add_argument("--headless", action="store_true", help="창 없이 실행")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0이면 창을 닫을 때까지 실행, 양수이면 해당 스텝 뒤 종료",
    )
    parser.add_argument("--width", type=int, default=1280, help="영상 가로 해상도")
    parser.add_argument("--height", type=int, default=720, help="영상 세로 해상도")
    parser.add_argument(
        "--frame-skip-count",
        type=int,
        default=1,
        help="0은 60 Hz, 1은 약 30 Hz 발행",
    )
    args, _unknown = parser.parse_known_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width와 height는 양수여야 합니다.")
    if args.frame_skip_count < 0:
        parser.error("frame-skip-count는 0 이상이어야 합니다.")
    return args


ARGS = parse_arguments()
SIMULATION_APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "sync_loads": True,
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

# Isaac Sim 모듈은 SimulationApp 생성 이후에 import해야 한다.
import omni.graph.core as og
import omni.usd
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import is_stage_loading
from pxr import Usd, UsdGeom


PROJECT_DIR = Path(__file__).resolve().parent
STAGE_PATH = PROJECT_DIR / "m0617_3fgripper08201638.usd"
CAMERA_ROOT_PATHS = {
    "robot_01": "/World/base_rsd455_01",
    "robot_02": "/World/base_rsd455_02",
}
CAMERA_ROOT_PATH = CAMERA_ROOT_PATHS[ARGS.robot_id]
CAMERA_PRIM_PATH = f"{CAMERA_ROOT_PATH}/RSD455/Camera_OmniVision_OV9782_Color"
GRAPH_PATH = f"/BaseCameraRosGraph_{ARGS.robot_id}"

# 이름은 harvest_namespace 가 정한다. 두 로봇을 동시에 띄우면 예전처럼 고정
# 이름을 쓰던 시절에는 두 카메라가 한 토픽에 발행하고 서로 다른 위치의 카메라가
# 같은 TF frame 을 주장해서, 수신 측이 어느 로봇의 영상인지 구분할 수 없었다.
NAMES = HarvestNames(ARGS.robot_id)
FRAME_ID = NAMES.camera_frame

RGB_TOPIC = NAMES.rgb_topic
DEPTH_TOPIC = NAMES.depth_topic
CAMERA_INFO_TOPIC = NAMES.camera_info_topic


def require_prim(stage, prim_path):
    """필수 Prim이 없으면 잘못된 Stage를 실행한 것으로 보고 중단한다."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Stage에서 Prim을 찾을 수 없습니다: {prim_path}")
    return prim


def open_project_stage():
    """프로젝트 USD를 열고 payload와 reference 로딩을 기다린다."""
    if not STAGE_PATH.exists():
        raise FileNotFoundError(STAGE_PATH)

    omni.usd.get_context().open_stage(str(STAGE_PATH))
    SIMULATION_APP.update()
    SIMULATION_APP.update()
    while is_stage_loading():
        SIMULATION_APP.update()

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        base_camera_prim = stage.GetPrimAtPath(CAMERA_ROOT_PATH)
        if base_camera_prim.IsValid():
            raise RuntimeError(
                f"{CAMERA_ROOT_PATH} Prim은 있지만 D455 payload 안의 Color Camera가 "
                "로드되지 않았습니다. 에셋 서버 연결 또는 로컬 캐시를 "
                "확인하세요. 예상 경로: "
                f"{CAMERA_PRIM_PATH}"
            )
        raise RuntimeError(
            f"프로젝트 Stage에 {CAMERA_ROOT_PATH}가 없습니다. 예상 Camera 경로: "
            f"{CAMERA_PRIM_PATH}"
        )

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if not np.isclose(meters_per_unit, 1.0):
        raise RuntimeError(f"Stage 단위가 meter가 아닙니다: {meters_per_unit}")
    return stage, camera_prim


def quaternion_multiply_xyzw(first, second):
    """ROS가 사용하는 [x, y, z, w] 순서의 quaternion을 곱한다."""
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )
    return result / np.linalg.norm(result)


def camera_optical_world_pose(stage, camera_prim):
    """ROS optical frame의 월드 pose를 USD 카메라 pose에서 계산한다.

    USD Camera는 +X가 오른쪽, +Y가 위쪽, -Z가 전방이다. ROS optical
    frame은 +X가 오른쪽, +Y가 아래쪽, +Z가 전방이므로 로컬 X축 기준
    180도 회전을 추가한다.
    """
    world_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        camera_prim
    )
    translation = np.array(world_matrix.ExtractTranslation(), dtype=float)
    usd_quat = world_matrix.ExtractRotationQuat()
    imag = usd_quat.GetImaginary()
    world_from_usd = np.array(
        [float(imag[0]), float(imag[1]), float(imag[2]), float(usd_quat.GetReal())],
        dtype=float,
    )
    usd_from_optical = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    world_from_optical = quaternion_multiply_xyzw(
        world_from_usd, usd_from_optical
    )
    return translation, world_from_optical


def create_ros_graph(stage, camera_prim):
    """저장 파일이 아닌 Session Layer에 ROS 2 발행 그래프를 만든다."""
    if stage.GetPrimAtPath(GRAPH_PATH).IsValid():
        raise RuntimeError(
            f"이미 ROS 카메라 그래프가 존재합니다: {GRAPH_PATH}. "
            "중복 실행 여부를 확인하세요."
        )

    translation, rotation = camera_optical_world_pose(stage, camera_prim)
    keys = og.Controller.Keys

    # Session Layer를 사용하므로 프로그램 종료 후 원본 USD에는 그래프가
    # 저장되지 않는다.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": GRAPH_PATH,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
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
                        "PublishBaseCameraTf",
                        "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                    ),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishClock.inputs:timeStamp",
                    ),
                    (
                        "OnPlaybackTick.outputs:tick",
                        "CreateRenderProduct.inputs:execIn",
                    ),
                    (
                        "CreateRenderProduct.outputs:execOut",
                        "PublishRgb.inputs:execIn",
                    ),
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
                    (
                        "OnPlaybackTick.outputs:tick",
                        "PublishBaseCameraTf.inputs:execIn",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishBaseCameraTf.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    (
                        "CreateRenderProduct.inputs:cameraPrim",
                        [usdrt.Sdf.Path(CAMERA_PRIM_PATH)],
                    ),
                    ("CreateRenderProduct.inputs:width", ARGS.width),
                    ("CreateRenderProduct.inputs:height", ARGS.height),
                    ("PublishRgb.inputs:frameId", FRAME_ID),
                    ("PublishRgb.inputs:topicName", RGB_TOPIC),
                    ("PublishRgb.inputs:type", "rgb"),
                    (
                        "PublishRgb.inputs:frameSkipCount",
                        ARGS.frame_skip_count,
                    ),
                    ("PublishDepth.inputs:frameId", FRAME_ID),
                    ("PublishDepth.inputs:topicName", DEPTH_TOPIC),
                    ("PublishDepth.inputs:type", "depth"),
                    (
                        "PublishDepth.inputs:frameSkipCount",
                        ARGS.frame_skip_count,
                    ),
                    ("PublishCameraInfo.inputs:frameId", FRAME_ID),
                    (
                        "PublishCameraInfo.inputs:topicName",
                        CAMERA_INFO_TOPIC,
                    ),
                    (
                        "PublishCameraInfo.inputs:frameSkipCount",
                        ARGS.frame_skip_count,
                    ),
                    ("PublishBaseCameraTf.inputs:topicName", "/tf_static"),
                    ("PublishBaseCameraTf.inputs:parentFrameId", "world"),
                    ("PublishBaseCameraTf.inputs:childFrameId", FRAME_ID),
                    ("PublishBaseCameraTf.inputs:staticPublisher", True),
                    (
                        "PublishBaseCameraTf.inputs:translation",
                        translation.tolist(),
                    ),
                    (
                        "PublishBaseCameraTf.inputs:rotation",
                        rotation.tolist(),
                    ),
                ],
            },
        )

    return translation, rotation


def main():
    """Stage를 재생하면서 센서와 simulation time을 지속 발행한다."""
    required_libraries = (
        "librmw_implementation.so",
        "libament_index_cpp.so",
        "librcutils.so",
        "librcpputils.so",
    )
    missing_libraries = [
        name for name in required_libraries if not (ROS2_BRIDGE_LIB_DIR / name).is_file()
    ]
    if missing_libraries:
        raise RuntimeError(
            "ROS 2 Bridge 필수 라이브러리가 없습니다: "
            + ", ".join(missing_libraries)
        )

    enable_extension("isaacsim.ros2.bridge")
    SIMULATION_APP.update()

    bridge_node_type = "isaacsim.ros2.bridge.ROS2PublishClock"
    if og.GraphRegistry().get_node_type_version(bridge_node_type) <= 0:
        raise RuntimeError(
            "ROS 2 Bridge가 정상적으로 시작되지 않았습니다. "
            f"LD_LIBRARY_PATH에 {ROS2_BRIDGE_LIB_DIR}가 포함되어 있는지와 "
            "Isaac Sim 로그의 'ROS2 Bridge startup failed' 메시지를 확인하세요."
        )

    stage, camera_prim = open_project_stage()
    translation, rotation = create_ros_graph(stage, camera_prim)
    SIMULATION_APP.update()

    simulation_context = SimulationContext(
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
    )
    simulation_context.initialize_physics()
    simulation_context.play()

    print("\n" + "=" * 72)
    print(f" {ARGS.robot_id} base_rsd455 ROS 2 Publisher")
    print("=" * 72)
    print(f" Stage       : {STAGE_PATH}")
    print(f" Robot ID    : {ARGS.robot_id}")
    print(f" Camera Root : {CAMERA_ROOT_PATH}")
    print(f" Camera Prim : {CAMERA_PRIM_PATH}")
    print(f" Resolution  : {ARGS.width} x {ARGS.height}")
    print(f" RGB         : {RGB_TOPIC}")
    print(f" Depth       : {DEPTH_TOPIC}")
    print(f" CameraInfo  : {CAMERA_INFO_TOPIC}")
    print(f" TF          : world -> {FRAME_ID}")
    print(f" TF position : {translation}")
    print(f" TF rotation : {rotation} (x, y, z, w)")

    step = 0
    try:
        while SIMULATION_APP.is_running():
            simulation_context.step(render=True)
            step += 1
            if ARGS.max_steps > 0 and step >= ARGS.max_steps:
                break
    except KeyboardInterrupt:
        print("\n사용자가 센서 발행을 중단했습니다.")
    finally:
        simulation_context.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {ARGS.robot_id} base_rsd455 발행 실패: {error}", file=sys.stderr)
        raise
    finally:
        SIMULATION_APP.close()
