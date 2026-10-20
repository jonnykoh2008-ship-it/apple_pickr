"""base_rsd455 RGB-D 영상에서 사과를 검출해 world target을 발행한다 (개인 PC 1).

이 파일은 Isaac Sim Python이 아니라 ROS 2 Jazzy가 설치된 시스템 환경에서
실행한다. 현재 단계의 HSV 검출은 RGB-D와 TF 파이프라인 검증용이며, 여러
품종과 조명 조건을 다룰 때는 검출기 부분을 학습 기반 segmentation으로
교체해야 한다.

멀티로봇 운용에서는 로봇마다 하나씩 띄운다. ``--robot-id`` 가 구독·발행
이름과 TF frame을 모두 결정하므로, 두 대를 동시에 띄워도 한 나무의 검출이
다른 로봇을 움직이지 않는다.

구독:
    /<robot_id>/base_camera/color/image_raw
    /<robot_id>/base_camera/depth/image_raw
    /<robot_id>/base_camera/camera_info
    /simulation/state
    /tf, /tf_static

발행:
    /<robot_id>/harvest/detection_pose_camera  카메라 좌표의 검출 결과
    /<robot_id>/harvest/target                 world 수확 목표
    /<robot_id>/harvest/perception_status      target 생성 전후의 인식 상태
    /<robot_id>/harvest/detection_debug        검출 윤곽이 표시된 RGB 영상

실행:
    source /opt/ros/jazzy/setup.bash
    ROS_DOMAIN_ID=102 python3 base_apple_detector.py --robot-id robot_01

use_sim_time은 노드가 직접 강제하므로 별도 인자가 필요하지 않다.
"""

import argparse
import math
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from appleproj_interfaces.msg import (
    HarvestPerceptionStatus,
    HarvestTarget,
    SimulationState,
)
from geometry_msgs.msg import PointStamped

from harvest_namespace import HarvestNames, add_robot_id_argument


# ══════════════════════════════════════════════════════════════
# 검출 파라미터
#
# 사과는 빨강이라 HSV의 Hue가 0도 근처에서 끊긴다. 한 구간으로 잡으면
# 색상환이 넘어가는 쪽 절반을 놓치므로 두 구간을 OR로 합친다.
# ══════════════════════════════════════════════════════════════
HSV_LOWER_1 = np.array([0, 90, 60], dtype=np.uint8)
HSV_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
HSV_LOWER_2 = np.array([170, 90, 60], dtype=np.uint8)
HSV_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA_PX = 150
MAX_CONTOUR_AREA_PX = 200000
# 사과는 둥글다. 윤곽 면적을 최소 외접원 면적으로 나눈 값이 이보다 작으면
# 사과가 아니라고 본다.
#
# 다만 나무에 달린 사과는 거의 항상 잎에 일부 가려져서, 온전한 원을 기대하면
# 안 된다. 실측에서 반경 42 px 로 또렷하게 보이는 사과가 0.505 로 나왔고
# 임계값 0.55 에 걸려 검출이 통째로 실패했다. 가늘고 긴 조각은 0.2 이하라
# 0.35 로 두면 사과만 남는다.
MIN_CIRCULARITY = 0.35

# D455 동작 범위. 이 밖의 depth는 신뢰하지 않는다.
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 6.00
# 검출 영역 안에서 유효한 depth 픽셀이 이 비율 미만이면 발행하지 않는다.
MIN_VALID_DEPTH_RATIO = 0.30

# RGB와 depth의 촬영 시각이 이보다 벌어지면 같은 순간으로 보지 않는다.
MAX_SYNC_DELTA_SEC = 0.05
# 영상 timestamp와 사용한 TF timestamp의 허용 차이. 통합 시험 전 임시값이다.
MAX_TF_TIME_ERROR_SEC = 0.20

MIN_CONFIDENCE = 0.30

# 명목 지름 80mm 사과의 시뮬레이션용 임시값. 기존 track의 마지막 world
# 위치에서 이 거리 안에 있는 최근접 후보만 같은 ID로 연결한다.
TRACK_MATCH_RADIUS_M = 0.100

# ID 집합을 확정하기 전에 새 사과가 나타나지 않아야 하는 연속 프레임 수.
# 약 30 Hz 발행 기준 2초에 해당한다. 로봇이 첫 Goal 을 받기까지 그보다
# 오래 걸리므로, 수확이 시작된 뒤에 새 ID 가 생기는 일은 없다.
FREEZE_STABLE_FRAMES = 60

WORLD_FRAME = "world"


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)
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


class AppleCandidate:
    """한 프레임에서 검출한 사과 하나."""

    __slots__ = (
        "pixel",
        "radius_px",
        "camera_point",
        "world_point",
        "confidence",
        "valid_depth_ratio",
        "contour",
        "target_id",
    )

    def __init__(self, pixel, radius_px, camera_point, confidence, valid_depth_ratio, contour):
        self.pixel = pixel
        self.radius_px = radius_px
        self.camera_point = camera_point
        self.world_point = None
        self.confidence = confidence
        self.valid_depth_ratio = valid_depth_ratio
        self.contour = contour
        self.target_id = ""


class AppleTracker:
    """같은 reset_id 안에서 사과 ID를 고정한다.

    ID 집합은 한 프레임이 아니라 관측 구간으로 만든다. 사과가 하나라도
    보이는 첫 프레임에서 곧바로 확정해 버리면, 그 순간 잎에 가려 있던
    사과는 영영 ID를 못 받아 target 으로 발행되지 않는다. 실제로 나무에
    사과가 남았는데 수확이 끝나 버리는 증상이 그것이었다.

    그래서 track 수가 ``FREEZE_STABLE_FRAMES`` 동안 늘지 않을 때까지 새
    사과를 계속 등록하고, 그 뒤에 확정한다. 확정 후에는 새 ID 를 만들지
    않으므로 수확되어 컨베이어로 간 사과가 다시 등록되지 않는다.
    """

    def __init__(
        self,
        match_radius_m=TRACK_MATCH_RADIUS_M,
        freeze_stable_frames=FREEZE_STABLE_FRAMES,
    ):
        self.match_radius_m = float(match_radius_m)
        self.freeze_stable_frames = int(freeze_stable_frames)
        self.tracks = {}
        self.frozen = False
        self.stable_frames = 0

    def reset(self):
        self.tracks.clear()
        self.frozen = False
        self.stable_frames = 0

    @staticmethod
    def _xyz_key(candidate):
        return (
            round(float(candidate.world_point[0]), 4),
            round(float(candidate.world_point[1]), 4),
            round(float(candidate.world_point[2]), 4),
        )

    def _match_existing(self, candidates):
        """기존 track 의 마지막 위치에서 반경 안의 최근접 후보를 잇는다."""
        unmatched = list(candidates)
        for target_id, last_position in list(self.tracks.items()):
            best = None
            for candidate in unmatched:
                distance = float(
                    np.linalg.norm(np.asarray(candidate.world_point) - last_position)
                )
                if distance > self.match_radius_m:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, candidate)
            if best is None:
                continue
            _, candidate = best
            candidate.target_id = target_id
            self.tracks[target_id] = np.asarray(candidate.world_point, dtype=float)
            unmatched.remove(candidate)
        return unmatched

    def assign(self, candidates):
        """후보에 target_id를 채운다. 연결되지 않은 후보는 빈 ID로 남는다."""
        if not candidates:
            return candidates

        unmatched = self._match_existing(candidates)

        if self.frozen:
            return candidates

        if unmatched:
            # 아직 확정 전이다. 새로 보인 사과를 등록하고 안정 카운터를 리셋한다.
            # 번호는 발견 순서로 이어 붙인다. 확정 시점에 다시 매기면 이미
            # 발행한 target_id 가 바뀌어 수신 측의 완료·대기 기록이 어긋난다.
            for candidate in sorted(unmatched, key=self._xyz_key):
                target_id = f"apple_{len(self.tracks) + 1:03d}"
                candidate.target_id = target_id
                self.tracks[target_id] = np.asarray(
                    candidate.world_point, dtype=float
                )
            self.stable_frames = 0
            return candidates

        self.stable_frames += 1
        if self.stable_frames >= self.freeze_stable_frames:
            self.frozen = True
        return candidates


class BaseAppleDetector(Node):
    """RGB-D에서 사과 중심을 계산해 world target을 발행한다."""

    def __init__(self, robot_id, publish_debug=True):
        self.names = HarvestNames(robot_id)
        super().__init__(f"base_apple_detector_{self.names.robot_id}")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.bridge = CvBridge()
        self.tracker = AppleTracker()
        self.publish_debug = bool(publish_debug)

        self.camera_info = None
        self.latest_rgb = None
        self.latest_depth = None
        self.simulation_state = None
        self.reset_id = 0
        self.scene_version = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Image, self.names.rgb_topic, self.on_rgb, SENSOR_QOS
        )
        self.create_subscription(
            Image, self.names.depth_topic, self.on_depth, SENSOR_QOS
        )
        self.create_subscription(
            CameraInfo, self.names.camera_info_topic, self.on_camera_info, SENSOR_QOS
        )
        self.create_subscription(
            SimulationState,
            self.names.simulation_state_topic,
            self.on_simulation_state,
            LATCHED_QOS,
        )

        self.target_publisher = self.create_publisher(
            HarvestTarget, self.names.target_topic, RELIABLE_QOS
        )
        self.status_publisher = self.create_publisher(
            HarvestPerceptionStatus, self.names.perception_status_topic, RELIABLE_QOS
        )
        self.camera_pose_publisher = self.create_publisher(
            PointStamped, self.names.detection_pose_camera_topic, RELIABLE_QOS
        )
        self.debug_publisher = (
            self.create_publisher(Image, self.names.detection_debug_topic, SENSOR_QOS)
            if self.publish_debug
            else None
        )

        self.get_logger().info("검출기 시작\n" + self.names.describe())

    # -- 입력 --------------------------------------------------------------
    def on_camera_info(self, message):
        self.camera_info = message

    def on_rgb(self, message):
        self.latest_rgb = message
        self.try_process()

    def on_depth(self, message):
        self.latest_depth = message

    def on_simulation_state(self, message):
        previous = self.simulation_state
        self.simulation_state = message
        if previous is None or message.reset_id != previous.reset_id:
            # Timeline reset. 이전 검출 캐시와 ID를 폐기한다.
            self.tracker.reset()
            self.get_logger().info(f"reset_id {message.reset_id} 감지, track 초기화")
        self.reset_id = message.reset_id
        self.scene_version = message.scene_version

    # -- 상태 보고 ---------------------------------------------------------
    def publish_status(
        self,
        status,
        header=None,
        target_id="",
        confidence=math.nan,
        valid_depth_ratio=math.nan,
        tf_time_error_sec=math.nan,
        message="",
    ):
        report = HarvestPerceptionStatus()
        if header is not None:
            report.header = header
        else:
            report.header.stamp = self.get_clock().now().to_msg()
            report.header.frame_id = self.names.camera_frame
        report.status = status
        report.target_id = target_id
        report.reset_id = self.reset_id
        report.scene_version = self.scene_version
        report.confidence = float(confidence)
        report.valid_depth_ratio = float(valid_depth_ratio)
        report.tf_time_error_sec = float(tf_time_error_sec)
        report.message = message
        self.status_publisher.publish(report)

    # -- 처리 --------------------------------------------------------------
    def try_process(self):
        rgb_message = self.latest_rgb
        depth_message = self.latest_depth

        if self.camera_info is None:
            self.publish_status(
                HarvestPerceptionStatus.INPUT_NOT_SYNCHRONIZED,
                message="CameraInfo를 아직 받지 못했습니다.",
            )
            return
        if rgb_message is None or depth_message is None:
            self.publish_status(
                HarvestPerceptionStatus.INPUT_NOT_SYNCHRONIZED,
                message="RGB 또는 depth 프레임이 없습니다.",
            )
            return

        delta_sec = abs(
            stamp_to_ns(rgb_message.header.stamp) - stamp_to_ns(depth_message.header.stamp)
        ) / 1e9
        if delta_sec > MAX_SYNC_DELTA_SEC:
            self.publish_status(
                HarvestPerceptionStatus.INPUT_NOT_SYNCHRONIZED,
                header=rgb_message.header,
                message=f"RGB-depth 시각 차이 {delta_sec * 1000:.1f} ms",
            )
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
        except Exception as error:  # noqa: BLE001
            self.publish_status(
                HarvestPerceptionStatus.INTERNAL_ERROR,
                header=rgb_message.header,
                message=f"이미지 변환 실패: {error}",
            )
            return

        depth = self.normalise_depth(depth)
        candidates = self.detect(rgb, depth)
        if self.debug_publisher is not None:
            self.publish_debug_image(rgb, candidates, rgb_message.header)

        if not candidates:
            self.publish_status(
                HarvestPerceptionStatus.NO_DETECTION,
                header=rgb_message.header,
                message="빨간 사과 윤곽을 찾지 못했습니다.",
            )
            return

        # 검출은 상태와 무관하게 계속하지만, READY/PLAYING이 아니면 target을
        # 발행하지 않는다.
        if self.simulation_state is None:
            self.publish_status(
                HarvestPerceptionStatus.SIMULATION_NOT_READY,
                header=rgb_message.header,
                message="/simulation/state를 아직 받지 못했습니다.",
            )
            return
        if self.simulation_state.state not in (
            SimulationState.READY,
            SimulationState.PLAYING,
        ):
            self.publish_status(
                HarvestPerceptionStatus.SIMULATION_NOT_READY,
                header=rgb_message.header,
                message=f"simulation state={self.simulation_state.state}",
            )
            return

        transform, tf_error_sec = self.lookup_camera_transform(rgb_message.header.stamp)
        if transform is None:
            self.publish_status(
                HarvestPerceptionStatus.TF_UNAVAILABLE,
                header=rgb_message.header,
                message=f"world <- {self.names.camera_frame} TF를 조회하지 못했습니다.",
            )
            return
        if tf_error_sec > MAX_TF_TIME_ERROR_SEC:
            self.publish_status(
                HarvestPerceptionStatus.STALE_FRAME,
                header=rgb_message.header,
                tf_time_error_sec=tf_error_sec,
                message=f"TF 시간 오차 {tf_error_sec * 1000:.1f} ms",
            )
            return

        valid = []
        for candidate in candidates:
            candidate.world_point = self.transform_point(transform, candidate.camera_point)
            if candidate.confidence < MIN_CONFIDENCE:
                continue
            if candidate.valid_depth_ratio < MIN_VALID_DEPTH_RATIO:
                continue
            valid.append(candidate)

        if not valid:
            worst = max(candidates, key=lambda item: item.confidence)
            status = (
                HarvestPerceptionStatus.DEPTH_INVALID
                if worst.valid_depth_ratio < MIN_VALID_DEPTH_RATIO
                else HarvestPerceptionStatus.LOW_CONFIDENCE
            )
            self.publish_status(
                status,
                header=rgb_message.header,
                confidence=worst.confidence,
                valid_depth_ratio=worst.valid_depth_ratio,
                tf_time_error_sec=tf_error_sec,
                message="유효한 후보가 없습니다.",
            )
            return

        self.tracker.assign(valid)

        published = 0
        for candidate in valid:
            if not candidate.target_id:
                # 최초 ID 집합이 만들어진 뒤에는 새 ID를 추가하지 않는다.
                continue
            self.publish_target(candidate, rgb_message.header, tf_error_sec)
            published += 1

        if published == 0:
            self.publish_status(
                HarvestPerceptionStatus.NO_DETECTION,
                header=rgb_message.header,
                tf_time_error_sec=tf_error_sec,
                message="기존 track에 연결된 후보가 없습니다.",
            )
            return

        best = max(valid, key=lambda item: item.confidence)
        self.publish_status(
            HarvestPerceptionStatus.OK,
            header=rgb_message.header,
            target_id=best.target_id,
            confidence=best.confidence,
            valid_depth_ratio=best.valid_depth_ratio,
            tf_time_error_sec=tf_error_sec,
            message=f"target {published}개 발행",
        )

    @staticmethod
    def normalise_depth(depth):
        """depth를 meter 단위 float32로 맞춘다.

        Isaac Sim은 32FC1 meter로 내지만, 16UC1 millimetre로 오는 경로도
        있어 단위를 확인한다.
        """
        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def detect(self, rgb, depth):
        """HSV 마스크와 윤곽으로 사과 후보를 만든다."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, HSV_LOWER_1, HSV_UPPER_1),
            cv2.inRange(hsv, HSV_LOWER_2, HSV_UPPER_2),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_CONTOUR_AREA_PX or area > MAX_CONTOUR_AREA_PX:
                continue
            (center_x, center_y), radius_px = cv2.minEnclosingCircle(contour)
            circle_area = math.pi * radius_px * radius_px
            if circle_area <= 0.0:
                continue
            circularity = area / circle_area
            if circularity < MIN_CIRCULARITY:
                continue

            depth_m, valid_ratio = self.sample_depth(depth, contour, mask.shape)
            if depth_m is None:
                continue
            camera_point = self.deproject(center_x, center_y, depth_m)
            if camera_point is None:
                continue
            # 원형에 가까울수록, 유효 depth가 많을수록 신뢰한다.
            confidence = float(np.clip(circularity * valid_ratio, 0.0, 1.0))
            candidates.append(
                AppleCandidate(
                    pixel=(float(center_x), float(center_y)),
                    radius_px=float(radius_px),
                    camera_point=camera_point,
                    confidence=confidence,
                    valid_depth_ratio=float(valid_ratio),
                    contour=contour,
                )
            )
        return candidates

    @staticmethod
    def sample_depth(depth, contour, shape):
        """윤곽 내부 depth의 중앙값과 유효 픽셀 비율.

        평균이 아니라 중앙값을 쓴다. 사과 가장자리에서 배경 depth가 섞이면
        평균은 뒤로 끌려가지만 중앙값은 표면 값을 유지한다.
        """
        region = np.zeros(shape[:2], dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
        values = depth[region == 255]
        if values.size == 0:
            return None, 0.0
        finite = values[np.isfinite(values)]
        usable = finite[(finite >= MIN_DEPTH_M) & (finite <= MAX_DEPTH_M)]
        ratio = float(usable.size) / float(values.size)
        if usable.size == 0:
            return None, ratio
        return float(np.median(usable)), ratio

    def deproject(self, pixel_x, pixel_y, depth_m):
        """픽셀과 depth를 카메라 광학 좌표계 3D 점으로 역투영한다."""
        info = self.camera_info
        if info is None:
            return None
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        x = (float(pixel_x) - cx) * depth_m / fx
        y = (float(pixel_y) - cy) * depth_m / fy
        return np.array([x, y, float(depth_m)], dtype=float)

    def lookup_camera_transform(self, stamp):
        """영상 timestamp에 가장 가까운 world <- camera 변환을 조회한다."""
        try:
            transform = self.tf_buffer.lookup_transform(
                WORLD_FRAME, self.names.camera_frame, Time.from_msg(stamp)
            )
        except TransformException:
            try:
                # 그 시각의 TF가 아직 없으면 최신 것으로 대체하고, 시간
                # 오차를 target에 그대로 실어 수신 측이 판단하게 한다.
                transform = self.tf_buffer.lookup_transform(
                    WORLD_FRAME, self.names.camera_frame, Time()
                )
            except TransformException:
                return None, math.inf
        error_sec = abs(
            stamp_to_ns(stamp) - stamp_to_ns(transform.header.stamp)
        ) / 1e9
        return transform, error_sec

    @staticmethod
    def transform_point(transform, point):
        """world <- camera 변환을 3D 점에 적용한다."""
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
        offset = np.array(
            [float(translation.x), float(translation.y), float(translation.z)],
            dtype=float,
        )
        return matrix @ np.asarray(point, dtype=float) + offset

    # -- 출력 --------------------------------------------------------------
    def publish_target(self, candidate, header, tf_error_sec):
        message = HarvestTarget()
        message.header.stamp = header.stamp
        message.header.frame_id = WORLD_FRAME
        message.target_id = candidate.target_id
        message.reset_id = self.reset_id
        message.scene_version = self.scene_version
        message.position.x = float(candidate.world_point[0])
        message.position.y = float(candidate.world_point[1])
        message.position.z = float(candidate.world_point[2])

        source = PointStamped()
        source.header.stamp = header.stamp
        source.header.frame_id = self.names.camera_frame
        source.point.x = float(candidate.camera_point[0])
        source.point.y = float(candidate.camera_point[1])
        source.point.z = float(candidate.camera_point[2])
        message.source_point = source

        message.confidence = float(candidate.confidence)
        message.valid_depth_ratio = float(candidate.valid_depth_ratio)
        message.tf_time_error_sec = float(tf_error_sec)

        self.target_publisher.publish(message)
        self.camera_pose_publisher.publish(source)

    def publish_debug_image(self, rgb, candidates, header):
        image = rgb.copy()
        for candidate in candidates:
            center = (int(candidate.pixel[0]), int(candidate.pixel[1]))
            cv2.drawContours(image, [candidate.contour], -1, (0, 255, 0), 2)
            cv2.circle(image, center, int(candidate.radius_px), (255, 200, 0), 1)
            label = candidate.target_id or "?"
            text = f"{label} {candidate.camera_point[2]:.2f}m c={candidate.confidence:.2f}"
            cv2.putText(
                image,
                text,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        try:
            debug = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        except Exception:  # noqa: BLE001 - 디버그 실패가 검출을 막지 않는다
            return
        debug.header = header
        self.debug_publisher.publish(debug)


def main():
    parser = argparse.ArgumentParser(description="base_rsd455 사과 검출기")
    add_robot_id_argument(parser)
    parser.add_argument(
        "--no-debug-image",
        action="store_true",
        help="검출 디버그 영상을 발행하지 않는다",
    )
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0], *remaining])
    node = None
    try:
        node = BaseAppleDetector(
            parsed.robot_id, publish_debug=not parsed.no_debug_image
        )
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
