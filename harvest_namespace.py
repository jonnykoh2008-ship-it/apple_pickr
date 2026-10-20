"""One place that decides how per-robot ROS 2 names are built.

The harvest stack was written for a single robot: every node published to
``/base_camera/...``, ``/harvest/target`` and ``/harvest/robot_motion``, and the
base camera TF frame was always ``base_camera``. ``--robot-id`` only picked
which USD prims to drive, so running two robots at once put two cameras on one
image topic, two Action servers on one Action name, and two differently placed
cameras claiming the same TF frame.

Nothing downstream could tell the two apart, so a detection from one tree could
send the other robot moving. Giving each robot its own namespace fixes that
without changing any message type.

The camera publisher, the Isaac harvest server and the detector all import from
here so a name can never drift between the node that publishes it and the node
that subscribes to it.
"""

from __future__ import annotations

ROBOT_IDS = ("robot_01", "robot_02")
DEFAULT_ROBOT_ID = "robot_01"


def normalise(robot_id: str) -> str:
    """Accept 'robot_01', '01' or '1' and return the canonical id."""
    text = str(robot_id).strip()
    if text in ROBOT_IDS:
        return text
    digits = text.lstrip("robot_").lstrip("0") or "0"
    candidate = f"robot_{int(digits):02d}"
    if candidate not in ROBOT_IDS:
        raise ValueError(
            f"unknown robot id {robot_id!r}; expected one of {ROBOT_IDS}"
        )
    return candidate


class HarvestNames:
    """Every ROS name one robot owns, derived from its id.

    Topics carry a leading slash and the robot id as the first segment. TF
    frames carry the same prefix without the slash, because a TF frame that
    starts with '/' is rejected by tf2 in ROS 2.
    """

    def __init__(self, robot_id: str = DEFAULT_ROBOT_ID) -> None:
        self.robot_id = normalise(robot_id)
        self.prefix = f"/{self.robot_id}"

    # -- camera -----------------------------------------------------------
    @property
    def rgb_topic(self) -> str:
        return f"{self.prefix}/base_camera/color/image_raw"

    @property
    def depth_topic(self) -> str:
        return f"{self.prefix}/base_camera/depth/image_raw"

    @property
    def camera_info_topic(self) -> str:
        return f"{self.prefix}/base_camera/camera_info"

    @property
    def camera_frame(self) -> str:
        """TF frame for the base camera. No leading slash: tf2 forbids it."""
        return f"{self.robot_id}/base_camera"

    # -- perception -------------------------------------------------------
    @property
    def target_topic(self) -> str:
        return f"{self.prefix}/harvest/target"

    @property
    def detection_debug_topic(self) -> str:
        return f"{self.prefix}/harvest/detection_debug"

    @property
    def detection_pose_camera_topic(self) -> str:
        return f"{self.prefix}/harvest/detection_pose_camera"

    @property
    def perception_status_topic(self) -> str:
        return f"{self.prefix}/harvest/perception_status"

    # -- motion -----------------------------------------------------------
    @property
    def robot_motion_action(self) -> str:
        return f"{self.prefix}/harvest/robot_motion"

    @property
    def motion_status_topic(self) -> str:
        return f"{self.prefix}/harvest/motion_status"

    @property
    def joint_states_topic(self) -> str:
        return f"{self.prefix}/joint_states"

    # -- shared across robots ---------------------------------------------
    # The conveyor, the simulation clock and the planning scene describe one
    # world, so they stay global on purpose. Namespacing them would give each
    # robot its own private copy of a thing there is only one of.
    simulation_state_topic = "/simulation/state"
    planning_scene_topic = "/planning_scene"
    planning_scene_service = "/planning_scene/get_snapshot"
    conveyor_checkpoint_topic = "/conveyor/checkpoint_events"
    conveyor_place_service = "/conveyor/place_command"
    conveyor_place_status_topic = "/conveyor/place_coordinator_status"

    def describe(self) -> str:
        return "\n".join([
            f" Robot ID    : {self.robot_id}",
            f" RGB         : {self.rgb_topic}",
            f" Depth       : {self.depth_topic}",
            f" CameraInfo  : {self.camera_info_topic}",
            f" Camera frame: {self.camera_frame}",
            f" Target      : {self.target_topic}",
            f" Motion      : {self.robot_motion_action}",
        ])


def add_robot_id_argument(parser, *, default: str = DEFAULT_ROBOT_ID):
    """Give a node the same --robot-id spelling as every other node."""
    parser.add_argument(
        "--robot-id",
        choices=ROBOT_IDS,
        default=default,
        help=f"ROS 이름과 USD prim을 고를 로봇 (기본값: {default})",
    )
    return parser
