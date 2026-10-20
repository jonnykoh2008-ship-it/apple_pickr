"""Publish three fixed conveyor RGB-D sensors through Isaac Sim ROS 2 Bridge.

Run only with the Isaac Sim 5.1.0 bundled Python 3.11.  The graph is authored
in the USD session layer, so the project stage is not modified on disk.

Published topics:
    /clock
    /conveyor_camera/color/image_raw
    /conveyor_camera/depth/image_raw
    /conveyor_camera/camera_info
    /conveyor_camera_01/{color/image_raw,depth/image_raw,camera_info}
    /conveyor_camera_02/{color/image_raw,depth/image_raw,camera_info}
    /tf_static  (world -> one optical frame per camera)
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# GPU PC 2의 품질 노드와 같은 domain이어야 토픽이 보인다. 현재 운용 domain은
# 103이다. trial_and_error/GPU_PC2_ROS_INTEGRATION_TEST.md는 아직 101로 적혀
# 있어 문서 갱신이 필요하다.
DEFAULT_ROS_DOMAIN_ID = "103"
os.environ.setdefault("ROS_DOMAIN_ID", DEFAULT_ROS_DOMAIN_ID)
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

ISAAC_SIM_ROOT = Path("/home/rokey/isaacsim")
ROS2_BRIDGE_LIB_DIR = ISAAC_SIM_ROOT / "exts/isaacsim.ros2.bridge/jazzy/lib"
ENV_REEXEC_GUARD = "CONVEYOR_CAMERA_PUBLISH_ENV_READY"


def ensure_ros2_bridge_library_path():
    if not ROS2_BRIDGE_LIB_DIR.is_dir():
        raise RuntimeError(
            "Isaac Sim ROS 2 Jazzy library directory is missing: "
            f"{ROS2_BRIDGE_LIB_DIR}"
        )
    paths = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value]
    bridge_path = str(ROS2_BRIDGE_LIB_DIR)
    if bridge_path in paths:
        return
    if os.environ.get(ENV_REEXEC_GUARD) == "1":
        raise RuntimeError(f"ROS 2 Bridge path was not applied: {bridge_path}")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join([bridge_path, *paths])
    environment[ENV_REEXEC_GUARD] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


ensure_ros2_bridge_library_path()

import numpy as np
from isaacsim import SimulationApp


# 탑 카메라를 광축 둘레로 돌리는 각도. 실측 근거는 다음과 같다. 컨베이어 면은
# 탑 카메라에서 0.442 m 떨어져 있고 벨트 폭은 0.379 m 인데, 스테이지의 자세로는
# 센서의 긴 축(1280, HFOV 90.5도)이 벨트를 가로지르고 짧은 축(720, VFOV 59.2도)
# 이 진행 방향을 향한다. 그래서 진행 방향으로는 0.502 m 만 담고, 가로로는 0.892 m
# 를 담아 그중 0.513 m 를 바닥에 쓴다.
#
# 90도 돌리면 두 축이 맞바뀌어 통과 구간이 0.502 m -> 0.892 m 로 78% 늘어난다.
# 카메라를 옮기지 않으므로 사과 지름은 115 px 그대로다. 0.4 m/s 에서 통과 시간이
# 1.25초에서 2.23초가 되고, 30Hz 에서 그룹이 37개에서 66개로 늘어 대표 순간
# 8개를 고를 여유가 생긴다. 벨트 폭 0.379 m 는 회전 후 세로 시야 0.502 m 안에
# 들어가므로 잘리지 않는다.
#
# 옆 카메라 두 대는 이미 긴 축이 진행 방향을 향하고 있어 회전이 필요 없다.
DEFAULT_TOP_CAMERA_ROLL_DEG = 0.0

# 세션 레이어에 추가하는 회전 op 의 이름. 스테이지 원본 op 와 구분하고 중복
# 적용을 막기 위해 쓴다.
CAMERA_ROLL_OP_SUFFIX = "conveyorRoll"


def parse_arguments():
    parser = argparse.ArgumentParser(description="conv_rsd455 ROS 2 publisher")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--rubber-roughness",
        type=float,
        default=0.9,
        help="runtime roughness override for materials bound to rubber prims",
    )
    parser.add_argument(
        "--rubber-specular",
        type=float,
        default=0.1,
        help="runtime specular override for materials bound to rubber prims",
    )
    parser.add_argument(
        "--frame-skip-count",
        type=int,
        default=1,
        help="0 publishes near 60 Hz; 1 publishes near 30 Hz",
    )
    parser.add_argument(
        "--top-camera-roll-deg",
        type=float,
        default=None,
        help=(
            "roll of the top camera about its own optical axis in degrees. "
            "The sensor is 1280x720 and the belt runs along image y, so the "
            "authored pose spends the long axis across a 0.38 m belt and puts "
            "only 0.50 m of conveyor in view; 90 swaps the axes for 0.89 m at "
            "the same resolution. Pass 0 to keep the authored pose "
            f"(default: {DEFAULT_TOP_CAMERA_ROLL_DEG})"
        ),
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help=(
            "ROS 2 domain for every published topic; must match the quality "
            f"node on GPU PC 2 (default: {DEFAULT_ROS_DOMAIN_ID})"
        ),
    )
    parser.add_argument(
        "--image-reliability",
        choices=("reliable", "bestEffort"),
        default="reliable",
        help=(
            "QoS reliability for RGB, depth and CameraInfo; use bestEffort "
            "only if retransmission latency breaks the result deadline"
        ),
    )
    parser.add_argument(
        "--image-queue-depth",
        type=int,
        default=6,
        help="QoS history depth for RGB, depth and CameraInfo",
    )
    parser.add_argument("--apple-static-friction", type=float, default=0.8)
    parser.add_argument("--apple-dynamic-friction", type=float, default=0.6)
    parser.add_argument("--apple-restitution", type=float, default=0.05)
    parser.add_argument(
        "--conveyor-speed",
        type=float,
        default=0.1,
        help=(
            "roller conveyor velocity in m/s; slower transport keeps the apple "
            "in the inspection ROI for more candidate frames; default: 0.1"
        ),
    )
    parser.add_argument(
        "--roller-collision-expansion",
        type=float,
        default=0.003,
        help=(
            "collision-only roller expansion in meters; prevents apples from "
            "settling into roller gaps without changing the visible meshes"
        ),
    )
    arguments, _unknown = parser.parse_known_args()
    if arguments.width <= 0 or arguments.height <= 0:
        parser.error("width and height must be positive")
    if arguments.max_steps < 0:
        parser.error("max-steps must be non-negative")
    if arguments.frame_skip_count < 0:
        parser.error("frame-skip-count must be non-negative")
    if not 0.0 <= arguments.rubber_roughness <= 1.0:
        parser.error("rubber-roughness must be between 0 and 1")
    if not 0.0 <= arguments.rubber_specular <= 1.0:
        parser.error("rubber-specular must be between 0 and 1")
    if arguments.apple_static_friction < 0.0:
        parser.error("apple-static-friction must be non-negative")
    if not 0.0 <= arguments.apple_dynamic_friction <= arguments.apple_static_friction:
        parser.error(
            "apple friction must satisfy 0 <= dynamic friction <= static friction"
        )
    if not 0.0 <= arguments.apple_restitution <= 1.0:
        parser.error("apple-restitution must be between 0 and 1")
    if arguments.conveyor_speed is not None and not math.isfinite(
        arguments.conveyor_speed
    ):
        parser.error("conveyor-speed must be finite")
    if not 0.0 <= arguments.roller_collision_expansion <= 0.01:
        parser.error("roller-collision-expansion must be between 0 and 0.01 m")
    if arguments.image_queue_depth < 1:
        parser.error("image-queue-depth must be positive")
    if arguments.ros_domain_id is not None:
        if not 0 <= arguments.ros_domain_id <= 232:
            parser.error("ros-domain-id must be between 0 and 232")
        os.environ["ROS_DOMAIN_ID"] = str(arguments.ros_domain_id)
    return arguments


def image_qos_profile(arguments):
    """QoS JSON accepted by ROS2CameraHelper/ROS2CameraInfoHelper.

    Keys mirror isaacsim.ros2.bridge OgnROS2QoSProfile so the graph nodes
    parse it without a separate profile node.
    """
    return json.dumps(
        {
            "history": "keepLast",
            "depth": int(arguments.image_queue_depth),
            "reliability": arguments.image_reliability,
            "durability": "volatile",
            "deadline": 0.0,
            "lifespan": 0.0,
            "liveliness": "systemDefault",
            "leaseDuration": 0.0,
        }
    )


ARGS = parse_arguments()
IMAGE_QOS_PROFILE = image_qos_profile(ARGS)
SIMULATION_APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "sync_loads": True,
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

# Isaac Sim modules must be imported after SimulationApp is created.
import omni.graph.core as og
import omni.usd
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import is_stage_loading
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


PROJECT_DIR = Path(__file__).resolve().parent
STAGE_PATH = PROJECT_DIR / "m0617_3fgripper08201638.usd"
CAMERA_CHILD_PATH = "RSD455/Camera_OmniVision_OV9782_Color"
TRANSPORT_SURFACE_PATH = "/World/RuntimeConveyorBeltSurface"
APPLE_PHYSICS_MATERIAL_PATH = "/World/RuntimeApplePhysicsMaterial"


@dataclass(frozen=True)
class CameraConfig:
    prim_name: str
    view_name: str
    topic_namespace: str
    frame_id: str
    graph_path: str
    # 광축 둘레 회전(도). 센서는 1280x720 이라 어느 쪽으로 눕혔는지가 컨베이어를
    # 얼마나 담는지를 결정한다. 0 은 스테이지에 있는 자세를 그대로 쓴다.
    roll_degrees: float = 0.0

    @property
    def rgb_topic(self):
        return f"{self.topic_namespace}/color/image_raw"

    @property
    def depth_topic(self):
        return f"{self.topic_namespace}/depth/image_raw"

    @property
    def camera_info_topic(self):
        return f"{self.topic_namespace}/camera_info"


ALL_CAMERA_CONFIGS = (
    CameraConfig(
        prim_name="conv_rsd455",
        view_name="top",
        topic_namespace="/conveyor_camera",
        frame_id="quality_camera_top_optical_frame",
        graph_path="/ConveyorCameraRosGraph",
        roll_degrees=DEFAULT_TOP_CAMERA_ROLL_DEG,
    ),
    CameraConfig(
        prim_name="conv_rsd455_01",
        view_name="left",
        topic_namespace="/conveyor_camera_01",
        frame_id="quality_camera_left_optical_frame",
        graph_path="/ConveyorCamera01RosGraph",
    ),
    CameraConfig(
        prim_name="conv_rsd455_02",
        view_name="right",
        topic_namespace="/conveyor_camera_02",
        frame_id="quality_camera_right_optical_frame",
        graph_path="/ConveyorCamera02RosGraph",
    ),
)

# 현재 스테이지에는 conv_rsd455(탑뷰) 한 대만 있다. 대수를 늘리려면 이 슬라이스를
# 넓히면 되고, 스테이지에 없는 카메라는 open_project_stage 가 건너뛴다.
CAMERA_CONFIGS = ALL_CAMERA_CONFIGS[:1]


def _camera_paths(config):
    for root in (f"/{config.prim_name}", f"/World/{config.prim_name}"):
        yield root, f"{root}/{CAMERA_CHILD_PATH}"


def find_conveyor_camera(stage, config):
    roots = {}
    for root_path, _camera_path in _camera_paths(config):
        root = stage.GetPrimAtPath(root_path)
        if root.IsValid():
            roots[str(root.GetPath())] = root
    for prim in stage.Traverse():
        if prim.GetName() == config.prim_name:
            roots[str(prim.GetPath())] = prim

    exact_matches = []
    color_matches = []
    for root_path, root in roots.items():
        exact_path = f"{root_path}/{CAMERA_CHILD_PATH}"
        exact = stage.GetPrimAtPath(exact_path)
        if exact.IsValid() and exact.IsA(UsdGeom.Camera):
            exact_matches.append((root_path, exact_path, exact))
            continue
        for descendant in Usd.PrimRange(root):
            if not descendant.IsA(UsdGeom.Camera):
                continue
            if "color" in descendant.GetName().lower():
                color_matches.append(
                    (root_path, str(descendant.GetPath()), descendant)
                )

    matches = exact_matches or color_matches
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        paths = [camera_path for _root, camera_path, _camera in matches]
        raise RuntimeError(
            f"multiple {config.prim_name} color cameras were found; "
            "remove the ambiguity: "
            f"{paths}"
        )

    expected = ", ".join(path for _root, path in _camera_paths(config))
    if roots:
        raise RuntimeError(
            f"{config.prim_name} root was found but no color Camera prim was loaded; "
            f"roots={sorted(roots)}, fixed candidates={expected}"
        )
    raise RuntimeError(
        f"{config.prim_name} is absent from the loaded stage; searched fixed candidates "
        f"and every prim name, fixed candidates={expected}"
    )


def open_project_stage():
    if not STAGE_PATH.is_file():
        raise FileNotFoundError(STAGE_PATH)
    omni.usd.get_context().open_stage(str(STAGE_PATH))
    SIMULATION_APP.update()
    SIMULATION_APP.update()
    while is_stage_loading():
        SIMULATION_APP.update()
    stage = omni.usd.get_context().get_stage()
    # 스테이지마다 카메라 대수가 다르다. 탑뷰 한 대만 있는 환경에서 나머지를
    # 찾다가 죽지 않도록, 없는 카메라는 건너뛰고 한 대도 없을 때만 실패한다.
    found = []
    for config in CAMERA_CONFIGS:
        try:
            found.append((config, *find_conveyor_camera(stage, config)))
        except RuntimeError as exc:
            print(f" [skip] {config.view_name} camera ({config.prim_name}): {exc}")
    if not found:
        raise RuntimeError(
            "no conveyor camera was found in the stage; expected one of "
            + ", ".join(c.prim_name for c in CAMERA_CONFIGS)
        )
    cameras = tuple(found)
    metres_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if not np.isclose(metres_per_unit, 1.0):
        raise RuntimeError(f"stage units must be metres, got {metres_per_unit}")
    return stage, cameras


def freeze_fixed_camera(stage, root_path):
    """Disable authored rigid bodies in the fixed D455 payload in session state."""

    disabled = 0
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root = stage.GetPrimAtPath(root_path)
        for prim in Usd.PrimRange(root):
            enabled = prim.GetAttribute("physics:rigidBodyEnabled")
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not enabled.IsValid():
                continue
            if enabled.IsValid():
                enabled.Set(False)
            else:
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            disabled += 1
        if disabled == 0:
            payload_root = stage.OverridePrim(f"{root_path}/RSD455")
            UsdPhysics.RigidBodyAPI.Apply(payload_root).CreateRigidBodyEnabledAttr(False)
            disabled = 1
    return disabled


def reduce_rubber_reflections(stage, roughness, specular):
    """Override bound rubber materials in the session layer only."""

    materials = {}
    for prim in stage.Traverse():
        if "rubber" not in str(prim.GetPath()).lower():
            continue
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        if material and material.GetPrim().IsValid():
            materials[str(material.GetPath())] = material

    updated = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for material_path, material in sorted(materials.items()):
            for prim in Usd.PrimRange(material.GetPrim()):
                if not prim.IsA(UsdShade.Shader):
                    continue
                shader = UsdShade.Shader(prim)
                for shader_input in shader.GetInputs():
                    name = shader_input.GetBaseName().lower()
                    type_name = str(shader_input.GetTypeName()).lower()
                    if type_name not in {"float", "double", "half"}:
                        continue
                    if "roughness" in name:
                        shader_input.Set(float(roughness))
                        updated.append((material_path, name, float(roughness)))
                    elif "specular" in name:
                        shader_input.Set(float(specular))
                        updated.append((material_path, name, float(specular)))
    return updated


def find_conveyor_graph(stage):
    """Locate the authored ConveyorBeltGraph without hard-coding its parent.

    The conveyor track prim is named by the authoring tool, so it may be
    ``ConveyorTrack``, ``ConveyorTrack_01`` and so on. Searching by node type
    keeps the publisher working when the scene is re-authored.
    """
    candidates = []
    for prim in stage.Traverse():
        if prim.GetName() != "ConveyorBeltGraph":
            continue
        node = prim.GetChild("ConveyorNode")
        if node.IsValid():
            candidates.append((prim, node))

    if not candidates:
        raise RuntimeError(
            "authored ConveyorBeltGraph with a ConveyorNode child was not found "
            "anywhere in the stage"
        )
    if len(candidates) > 1:
        paths = [str(graph.GetPath()) for graph, _node in candidates]
        raise RuntimeError(f"multiple ConveyorBeltGraph prims found: {paths}")

    graph, node = candidates[0]
    print(f" Conveyor    : {graph.GetPath()}")
    return graph, node


def configure_roller_apple_contact(
    stage,
    static_friction,
    dynamic_friction,
    restitution,
    conveyor_speed,
):
    """Use the authored roller conveyor and bind friction to apple colliders."""

    graph, conveyor_node = find_conveyor_graph(stage)
    targets = [
        str(path)
        for path in conveyor_node.GetRelationship("inputs:conveyorPrim").GetTargets()
    ]
    if not targets:
        raise RuntimeError(
            f"{conveyor_node.GetPath()} has no inputs:conveyorPrim target; "
            "author the belt or roller prim the conveyor should drive"
        )
    # The conveyor prim usually comes from a referenced Omniverse asset, so the
    # local layer holds only an `over` for it. Judge the composed stage rather
    # than the local specifier, and require geometry to actually be present.
    missing = [
        path
        for path in targets
        if not stage.GetPrimAtPath(path).IsValid()
        or not any(
            child.IsA(UsdGeom.Xformable)
            for child in Usd.PrimRange(stage.GetPrimAtPath(path))
        )
    ]
    if missing:
        raise RuntimeError(
            f"{conveyor_node.GetPath()} points at prims with no loaded geometry: "
            f"{missing}. The conveyor asset is referenced from Omniverse content, "
            "so check that the reference resolved (network or local cache) before "
            "re-authoring the stage."
        )

    surface = stage.GetPrimAtPath(TRANSPORT_SURFACE_PATH)
    if not surface.IsValid():
        raise RuntimeError(f"runtime transport surface is missing: {TRANSPORT_SURFACE_PATH}")

    bound_colliders = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # The continuous plane was useful for straight transport but prevented
        # the roller contact from rotating the apple.  Keep it out of physics.
        collision = surface.GetAttribute("physics:collisionEnabled")
        if collision.IsValid():
            collision.Set(False)
        else:
            UsdPhysics.CollisionAPI.Apply(surface).CreateCollisionEnabledAttr(False)

        material = UsdShade.Material.Define(stage, APPLE_PHYSICS_MATERIAL_PATH)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(float(static_friction)).Set(
            float(static_friction)
        )
        physics_material.CreateDynamicFrictionAttr(float(dynamic_friction)).Set(
            float(dynamic_friction)
        )
        physics_material.CreateRestitutionAttr(float(restitution)).Set(
            float(restitution)
        )

        all_prims = tuple(stage.TraverseAll())
        for rigid_prim in all_prims:
            if "apple" not in str(rigid_prim.GetPath()).lower():
                continue
            if not rigid_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            kinematic = rigid_prim.GetAttribute("physics:kinematicEnabled").Get()
            if bool(kinematic):
                continue
            rigid_path = rigid_prim.GetPath()
            for collider in all_prims:
                if not collider.GetPath().HasPrefix(rigid_path):
                    continue
                collision = collider.GetAttribute("physics:collisionEnabled")
                if not collider.HasAPI(UsdPhysics.CollisionAPI) and not collision.IsValid():
                    continue
                binding = UsdShade.MaterialBindingAPI(collider)
                if not binding:
                    binding = UsdShade.MaterialBindingAPI.Apply(collider)
                binding.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bound_colliders.append(str(collider.GetPath()))

    if not bound_colliders:
        raise RuntimeError("no dynamic apple collision prims accepted the physics material")
    velocity = graph.GetAttribute("graph:variable:Velocity")
    if not velocity.IsValid():
        raise RuntimeError("roller ConveyorBeltGraph Velocity variable is missing")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if conveyor_speed is not None:
            velocity.Set(float(conveyor_speed))
    speed = velocity.Get()
    if speed is None:
        raise RuntimeError("roller ConveyorBeltGraph Velocity has no value")
    # The ConveyorNode already names the prim it drives, so the roller path is
    # derived from the scene rather than assumed.
    return bound_colliders, float(speed), targets[0]


def expand_roller_collision(stage, expansion, roller_path):
    """Add a collision-only skin so apples cannot settle between rollers."""

    rollers = stage.GetPrimAtPath(roller_path)
    if not rollers.IsValid() or not rollers.HasAPI(UsdPhysics.CollisionAPI):
        # The expansion only closes gaps between rollers. A flat belt has no
        # gaps, so transport still works without it.
        print(
            f" Roller skin : skipped, {roller_path} has no collision API "
            "(expected for a flat belt)"
        )
        return None, None

    rest_offset = rollers.GetAttribute("physxCollision:restOffset")
    contact_offset = rollers.GetAttribute("physxCollision:contactOffset")
    if not rest_offset.IsValid() or not contact_offset.IsValid():
        print(
            f" Roller skin : skipped, {roller_path} has no PhysX collision "
            "offsets to expand"
        )
        return None, None

    # Rest offset changes only the effective collision surface.  Keep the
    # contact envelope slightly larger so PhysX creates contacts before the
    # expanded surfaces reach their resting separation.
    contact_margin = 0.002
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        rest_offset.Set(float(expansion))
        contact_offset.Set(float(expansion + contact_margin))
    return float(expansion), float(expansion + contact_margin)


def quaternion_multiply_xyzw(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    result = np.asarray(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dtype=float,
    )
    norm = np.linalg.norm(result)
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("camera quaternion is invalid")
    return result / norm


def roll_camera_about_optical_axis(stage, camera_prim, degrees):
    """Rotate a camera about its own view direction, in the session layer.

    A USD camera looks down its local -Z with +X right and +Y up, so a rotation
    about local Z is exactly a roll: it turns the sensor in its own plane
    without moving the camera or changing its intrinsics.

    The op must come last in xformOpOrder, which is where AddRotateZOp puts it.
    Measured on a camera placed at (0.15, 0.10, 0.442): appended, the camera
    stays put and only the sensor axes turn; moved to the front of the order it
    orbits the parent origin and ends up 25.5 cm away, pointing at a different
    stretch of belt. A camera authored directly overhead cannot tell the two
    apart, because a Z translation commutes with a Z rotation, so the mistake
    would stay hidden until a camera with a sideways offset was rolled.

    The edit goes to the session layer, like the ROS graph, so the project
    stage on disk is not modified.

    Returns the angle applied, or None when nothing was changed.
    """
    if not degrees:
        return None
    xformable = UsdGeom.Xformable(camera_prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName().endswith(f":{CAMERA_ROLL_OP_SUFFIX}"):
            raise RuntimeError(
                f"{camera_prim.GetPath()} already carries a "
                f"{CAMERA_ROLL_OP_SUFFIX} op; the stage was edited twice"
            )
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        roll = xformable.AddRotateZOp(
            UsdGeom.XformOp.PrecisionDouble, CAMERA_ROLL_OP_SUFFIX
        )
        roll.Set(float(degrees))
    return float(degrees)


def camera_optical_world_pose(camera_prim):
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        camera_prim
    )
    translation = np.asarray(matrix.ExtractTranslation(), dtype=float)
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    world_from_usd = np.asarray(
        (
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(quaternion.GetReal()),
        ),
        dtype=float,
    )
    # USD camera: +X right, +Y up, -Z forward. ROS optical: +X right,
    # +Y down, +Z forward. Rotate 180 degrees around local X.
    world_from_optical = quaternion_multiply_xyzw(
        world_from_usd, np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float)
    )
    return translation, world_from_optical


def create_ros_graph(stage, config, camera_path, camera_prim, *, publish_clock):
    if stage.GetPrimAtPath(config.graph_path).IsValid():
        raise RuntimeError(f"ROS camera graph already exists: {config.graph_path}")
    translation, rotation = camera_optical_world_pose(camera_prim)
    keys = og.Controller.Keys
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("PublishDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("PublishCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ("PublishTf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
        ("CreateRenderProduct.outputs:execOut", "PublishRgb.inputs:execIn"),
        ("CreateRenderProduct.outputs:execOut", "PublishDepth.inputs:execIn"),
        ("CreateRenderProduct.outputs:execOut", "PublishCameraInfo.inputs:execIn"),
        ("CreateRenderProduct.outputs:renderProductPath", "PublishRgb.inputs:renderProductPath"),
        ("CreateRenderProduct.outputs:renderProductPath", "PublishDepth.inputs:renderProductPath"),
        ("CreateRenderProduct.outputs:renderProductPath", "PublishCameraInfo.inputs:renderProductPath"),
        ("OnPlaybackTick.outputs:tick", "PublishTf.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishTf.inputs:timeStamp"),
    ]
    if publish_clock:
        nodes.append(("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"))
        connections.extend(
            (
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            )
        )
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {
                "graph_path": config.graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: nodes,
                keys.CONNECT: connections,
                keys.SET_VALUES: [
                    ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                    ("CreateRenderProduct.inputs:width", ARGS.width),
                    ("CreateRenderProduct.inputs:height", ARGS.height),
                    ("PublishRgb.inputs:frameId", config.frame_id),
                    ("PublishRgb.inputs:topicName", config.rgb_topic),
                    ("PublishRgb.inputs:type", "rgb"),
                    ("PublishRgb.inputs:frameSkipCount", ARGS.frame_skip_count),
                    ("PublishRgb.inputs:qosProfile", IMAGE_QOS_PROFILE),
                    ("PublishDepth.inputs:frameId", config.frame_id),
                    ("PublishDepth.inputs:topicName", config.depth_topic),
                    ("PublishDepth.inputs:type", "depth"),
                    ("PublishDepth.inputs:frameSkipCount", ARGS.frame_skip_count),
                    ("PublishDepth.inputs:qosProfile", IMAGE_QOS_PROFILE),
                    ("PublishCameraInfo.inputs:frameId", config.frame_id),
                    ("PublishCameraInfo.inputs:topicName", config.camera_info_topic),
                    ("PublishCameraInfo.inputs:frameSkipCount", ARGS.frame_skip_count),
                    ("PublishCameraInfo.inputs:qosProfile", IMAGE_QOS_PROFILE),
                    ("PublishTf.inputs:topicName", "/tf_static"),
                    ("PublishTf.inputs:parentFrameId", "world"),
                    ("PublishTf.inputs:childFrameId", config.frame_id),
                    ("PublishTf.inputs:staticPublisher", True),
                    ("PublishTf.inputs:translation", translation.tolist()),
                    ("PublishTf.inputs:rotation", rotation.tolist()),
                ],
            },
        )
    return translation, rotation


def validate_bridge():
    required = (
        "librmw_implementation.so",
        "libament_index_cpp.so",
        "librcutils.so",
        "librcpputils.so",
    )
    missing = [name for name in required if not (ROS2_BRIDGE_LIB_DIR / name).is_file()]
    if missing:
        raise RuntimeError("ROS 2 Bridge libraries are missing: " + ", ".join(missing))
    # The authored ConveyorBeltGraph stores its Velocity variable in USD, but
    # the graph cannot apply surface velocity until the native IsaacConveyor
    # node type is registered by this extension.
    enable_extension("isaacsim.asset.gen.conveyor")
    enable_extension("isaacsim.ros2.bridge")
    SIMULATION_APP.update()
    SIMULATION_APP.update()
    conveyor_node_type = "isaacsim.asset.gen.conveyor.IsaacConveyor"
    if og.GraphRegistry().get_node_type_version(conveyor_node_type) <= 0:
        raise RuntimeError(
            "Isaac Conveyor extension loaded without registering "
            f"{conveyor_node_type}"
        )
    bridge_node_type = "isaacsim.ros2.bridge.ROS2PublishClock"
    if og.GraphRegistry().get_node_type_version(bridge_node_type) <= 0:
        raise RuntimeError("Isaac Sim ROS 2 Bridge failed to start")


def main():
    validate_bridge()
    stage, cameras = open_project_stage()
    rigid_bodies_disabled = sum(
        freeze_fixed_camera(stage, root_path)
        for _config, root_path, _camera_path, _camera in cameras
    )
    (
        apple_material_colliders,
        conveyor_speed,
        roller_path,
    ) = configure_roller_apple_contact(
        stage,
        ARGS.apple_static_friction,
        ARGS.apple_dynamic_friction,
        ARGS.apple_restitution,
        ARGS.conveyor_speed,
    )
    roller_rest_offset, roller_contact_offset = expand_roller_collision(
        stage,
        ARGS.roller_collision_expansion,
        roller_path,
    )
    rubber_overrides = reduce_rubber_reflections(
        stage,
        ARGS.rubber_roughness,
        ARGS.rubber_specular,
    )
    # 회전은 그래프 생성 전에 끝내야 한다. create_ros_graph 가 prim 의
    # local-to-world 를 읽어 TF static 으로 굽고, RGB/depth/CameraInfo 는 모두
    # 같은 render product 에서 나오므로 이 시점에 돌려 두면 넷이 함께 따라온다.
    camera_rolls = []
    for config, _root_path, _camera_path, camera in cameras:
        degrees = config.roll_degrees
        if config.view_name == "top" and ARGS.top_camera_roll_deg is not None:
            degrees = ARGS.top_camera_roll_deg
        applied = roll_camera_about_optical_axis(stage, camera, degrees)
        if applied is not None:
            camera_rolls.append((config.view_name, applied))
    if camera_rolls:
        SIMULATION_APP.update()

    published_cameras = []
    for index, (config, _root_path, camera_path, camera) in enumerate(cameras):
        translation, rotation = create_ros_graph(
            stage,
            config,
            camera_path,
            camera,
            publish_clock=index == 0,
        )
        published_cameras.append((config, camera_path, translation, rotation))
    SIMULATION_APP.update()

    simulation = SimulationContext(
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
    )
    simulation.initialize_physics()
    simulation.play()
    print("\n" + "=" * 72)
    print(" Conveyor three-view RGB-D ROS 2 Publisher")
    print("=" * 72)
    print(f" Stage       : {STAGE_PATH}")
    for config, camera_path, _translation, _rotation in published_cameras:
        print(f" {config.view_name:>5} camera: {camera_path}")
    print(f" Frozen body : {rigid_bodies_disabled}")
    print(f" Conveyor    : authored rollers at {conveyor_speed:.3f} m/s")
    if roller_rest_offset is not None:
        print(
            " Roller skin : "
            f"rest={roller_rest_offset:.3f} m "
            f"contact={roller_contact_offset:.3f} m"
        )
    print(f" Flat surface: {TRANSPORT_SURFACE_PATH} collision disabled")
    print(
        " Apple mat.  : "
        f"static={ARGS.apple_static_friction} "
        f"dynamic={ARGS.apple_dynamic_friction} "
        f"restitution={ARGS.apple_restitution} "
        f"colliders={len(apple_material_colliders)}"
    )
    print(f" Rubber attrs: {len(rubber_overrides)} runtime overrides")
    for material_path, input_name, value in rubber_overrides:
        print(f"               {material_path}.{input_name}={value}")
    print(f" Resolution  : {ARGS.width} x {ARGS.height}")
    if camera_rolls:
        for view_name, degrees in camera_rolls:
            print(
                f" Camera roll : {view_name} rolled {degrees:g} deg about its "
                "optical axis (long sensor axis along the conveyor)"
            )
    else:
        print(" Camera roll : none; authored poses are used as they are")
    # GPU PC 2가 구독 계약을 눈으로 확인할 수 있도록 실제 적용값을 남긴다.
    print(
        " ROS 2 env   : "
        f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')} "
        f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION')}"
    )
    print(f" Image QoS   : {IMAGE_QOS_PROFILE}")
    print(
        " Depth       : DistanceToImagePlane (optical Z-depth), "
        "sensor_msgs/Image 32FC1 metres, "
        "aligned to RGB via the shared render product "
        "(identical resolution and intrinsics); "
        "invalid pixels are 0 or non-finite"
    )
    publish_hz = 60.0 / float(ARGS.frame_skip_count + 1)
    print(f" Publish rate: ~{publish_hz:.0f} Hz (frame-skip {ARGS.frame_skip_count})")
    for config, _camera_path, translation, rotation in published_cameras:
        print(f" [{config.view_name}] RGB        : {config.rgb_topic}")
        print(f" [{config.view_name}] Depth      : {config.depth_topic}")
        print(f" [{config.view_name}] CameraInfo : {config.camera_info_topic}")
        print(f" [{config.view_name}] TF         : world -> {config.frame_id}")
        print(f" [{config.view_name}] position   : {translation}")
        print(f" [{config.view_name}] rotation   : {rotation} (x, y, z, w)")

    step = 0
    try:
        while SIMULATION_APP.is_running():
            simulation.step(render=True)
            step += 1
            if ARGS.max_steps > 0 and step >= ARGS.max_steps:
                break
    except KeyboardInterrupt:
        print("\nConveyor camera publisher interrupted")
    finally:
        simulation.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] conv_rsd455 publisher failed: {error}", file=sys.stderr)
        raise
    finally:
        SIMULATION_APP.close()
