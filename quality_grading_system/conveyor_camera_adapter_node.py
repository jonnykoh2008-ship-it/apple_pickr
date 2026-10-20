"""Adapt synchronized conveyor RGB-D topics to the quality inspection contract."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

from inspection_session import (
    INSPECTION_ROI_FRAME,
    MAX_REPRESENTATIVE_FRAMES,
    MIN_INSTANT_GAP_NS,
    QUALITY_CAMERA_OPTICAL_FRAMES,
    REPRESENTATIVE_INSTANTS,
)
from opencv_size_grader import AppleNotDetected, DetectionConfig, detect_single_apple


# 컨베이어 2의 고정 카메라 3대. 한 사과의 서로 다른 면을 동시에 보므로 세 view를
# 하나의 검사로 묶어 표면을 덮는다. 손상은 한쪽 면에만 나타날 수 있다.
KNOWN_CAMERA_NAMESPACES = (
    "/conveyor_camera",
    "/conveyor_camera_01",
    "/conveyor_camera_02",
)

# 현재 스테이지에는 탑뷰 한 대만 있다. 표면 커버리지는 동시 관측이 아니라 통과
# 중 회전에서 나오므로 한 대로도 성립한다: 세 대를 동시에 써도 한 순간에 덮이는
# 것은 표면의 36.2% 이고(세 대가 모두 위를 보아 크게 겹친다), 탑뷰 한 대로 순간
# 여럿을 모으면 실측 ground truth 79.5% 대비 평균 편차 +0.1pp, 표준편차 2.0pp 로
# 수렴한다. 카메라를 늘리려면 이 튜플에 namespace 를 추가하면 된다.
DEFAULT_CAMERA_NAMESPACES = KNOWN_CAMERA_NAMESPACES[:1]
RGB_SUFFIX = "/color/image_raw"
DEPTH_SUFFIX = "/depth/image_raw"
CAMERA_INFO_SUFFIX = "/camera_info"

INSPECTION_TOPIC = "/quality/inspection_images"
COMPLETION_TOPIC = "/quality/inspection_completed"
DETECTION_DEBUG_TOPIC = "/quality/conveyor_detection_debug/compressed"
DEFAULT_DETECTION_DEBUG_SCALE = 0.5
DEFAULT_DETECTION_DEBUG_JPEG_QUALITY = 75
DEFAULT_REARM_ABSENT_FRAMES = 3
DEFAULT_SYNC_QUEUE_SIZE = 20

# 롤러 위에서 사과가 회전하면 정반사 면은 채도가 30 아래로 내려가고 밝기는
# 255까지 올라간다. 정지 영상용 기본값(35, 250)을 그대로 쓰면 같은 사과가
# 회전할 때마다 사라지므로 adapter 기본값은 실측 범위를 포함한다.
DEFAULT_ROLLING_MIN_SATURATION = 20
DEFAULT_ROLLING_MAX_VALUE = 255
ROLLING_FALLBACK_MIN_SATURATION = 8
ROLLING_FALLBACK_HUE_LOW_MAX = 35
ROLLING_FALLBACK_BBOX_EXPANSION = 0.75
ROLLING_FALLBACK_MIN_DIAMETER_RATIO = 0.55
ROLLING_FALLBACK_MAX_DIAMETER_RATIO = 1.60

# 검사 ROI는 카메라 영상 안의 세로 띠로 정의한다. docs/features/conveyor.md는
# 프레임 수집의 시작과 종료를 trigger collider가 아니라 카메라 ROI로 판단하도록
# 규정한다. 값은 영상 너비에 대한 비율이며 초기 시험값이다.
#
# 이 세로 띠 판정은 카메라마다 다른 물리량을 잰다. 수직 하방을 보는 top 은 사과
# 진행이 image y 로 나타나므로 image x 는 좌우 흔들림만 재고(실측 변동폭 0.184),
# 45도로 보는 left/right 는 image x 가 진행을 잰다(0.784, 0.723). 세 카메라에
# 같은 조건을 걸면 서로 다른 질문을 AND 로 묶게 되어, 사과가 검사 지점에
# 없는데도 세션이 열리고 금방 끊긴다. 아래 3D 판정이 기본이며 이 값들은
# depth 를 못 쓸 때의 대비책으로만 남는다.
DEFAULT_ROI_MIN_X_RATIO = 0.25
DEFAULT_ROI_MAX_X_RATIO = 0.75

# 3D ROI. 카메라 세 대는 모두 검사 지점을 조준하고 있으므로, 그 지점은 각
# 카메라의 광학 좌표계에서 광축 위 DEFAULT_ROI_AIM_DISTANCE_M 앞에 있다. 사과를
# depth 로 역투영해 그 지점과의 거리를 재면 세 카메라가 같은 물리량을 판정하고,
# TF 없이도 성립하며, 카메라를 옮겨도 조준 거리만 갱신하면 된다.
DEFAULT_ROI_AIM_DISTANCE_M = 0.40
# 검사 지점을 중심으로 한 허용 반경. 사과 지름이 약 0.08m 이므로 그 정도면
# 사과가 지점을 지나는 구간을 담되 상류의 사과는 배제한다.
DEFAULT_ROI_RADIUS_M = 0.08

# ROI 판정 방식. "aim_sphere" 는 조준점 주변 구로 제한하는 기존 동작이고,
# "full_frame" 은 카메라가 보는 컨베이어 구간 전체를 검사 구간으로 쓴다.
#
# 조준거리 0.40m 에서 탑 카메라의 가로 시야는 약 0.75m 인데(D455 HFOV 86도
# 기준), 반지름 0.08m 구는 그중 0.16m 만 쓴다. 시야의 79% 를 버리는 셈이다.
# 실제 운용 속도 0.3~0.4 m/s 에서 구 통과는 0.40~0.53초라 0.35초 간격으로는
# 순간 2개밖에 못 모으고, 커버리지가 목표 99.6% 가 아니라 약 55% 에 그친다.
# 시야 전체를 쓰면 통과가 1.87초로 늘어 30Hz 에서 55 그룹이 들어온다.
#
# 다만 탑뷰만으로는 한계가 있다. 사과가 벨트 가로 방향 축으로 구르면 그 축의
# 양 극점은 굴러도 제자리라, 순간을 무한히 늘려도 탑뷰 커버리지는 86.5% 에서
# 멈춘다. 나머지 13.5% 는 옆 카메라만 볼 수 있다. 그래서 시야를 넓히는 것과
# 카메라를 줄이는 것은 별개의 결정이다.
#
# 기본값은 기존 동작을 유지한다. 전환 판단에 필요한 시야 폭과 사과 간격은
# 아래 진단 로그로 실주행에서 측정한다.
ROI_MODE_AIM_SPHERE = "aim_sphere"
ROI_MODE_FULL_FRAME = "full_frame"
# 실측으로 aim_sphere 는 이 컨베이어에서 쓸 수 없다는 것이 확인되어 기본값을
# 옮겼다. 0.4 m/s 에서 반지름 0.08m 구는 0.40초에 통과되고 0.35초 간격으로는
# 순간이 2개밖에 모이지 않아 커버리지가 약 55% 에 그친다. 시야 전체를 쓰면
# 같은 속도에서 통과가 2.23초가 된다. aim_sphere 는 조준점을 좁게 보고 싶을 때
# 남겨 둔다.
DEFAULT_ROI_MODE = ROI_MODE_FULL_FRAME

# full_frame 에서 화면 가장자리에 걸린 사과는 실루엣이 잘려 직경도 착색률도
# 없는 표면을 기준으로 재게 된다. 경계에 닿은 사과는 제외한다.
DEFAULT_ROI_EDGE_MARGIN_PX = 16

# apple_id tracker. docs/features/conveyor.md 는 GPU PC 2 가 apple_id tracker 를
# 상시 수행하도록 규정하지만, 구현은 세션이 열릴 때마다 새 id 를 만들고 있었다.
# 그래서 검출이 한 번 끊겨 세션이 다시 열리면 같은 사과가 새 사과가 되었고,
# 실측에서 사과 하나가 검사 6건으로 갈렸다.
#
# 추적은 3D 위치의 연속성으로 한다. 세션이 끝난 뒤 이 시간 안에, 마지막으로 본
# 위치에서 이 반경 안에 사과가 다시 나타나면 같은 개체로 보고 apple_id 를
# 재사용한다. 사과는 초당 약 0.06m 로 움직이므로 1.5초면 0.09m 를 지난다.
DEFAULT_TRACK_GAP_NS = 1_500_000_000
DEFAULT_TRACK_RADIUS_M = 0.12
# ROI 안에서 사과를 놓친 그룹이 이만큼 연속되면 이탈로 확정한다. 한두 프레임의
# 검출 실패로 검사가 조기 종료되는 것을 막는다.
DEFAULT_ROI_EXIT_PATIENCE = 3
# ROI 경계에서 검출이 깜빡이면 이탈 직후 후보 한두 개짜리 허수 세션이 열려
# 3면을 못 채우고 TIMEOUT 으로 끝난다. 연속 진입을 요구해 그것을 막는다.
DEFAULT_ROI_ENTRY_PATIENCE = 2

# 세션을 유지하려면 몇 면이 동시에 ROI 안에 있어야 하는지.
#
# 이 값은 원래 카메라 수(3)였다. ROI 를 카메라별 이미지 밴드로 잡던 시절에는
# 한 면만 요구하면 사과가 세 밴드를 차례로 지나며 세션이 조각났기 때문이다.
# full_frame 은 밴드를 쓰지 않으므로 그 이유가 사라졌고, 3면 요구는 오히려
# 검사를 아예 못 열게 만든다. 실측 895 프레임에서 검출률은 탑 7.7%, 좌 16.5%,
# 우 15.1% 였고 세 면이 동시에 잡힌 그룹은 in_roi=0, 즉 한 번도 없었다.
# 탑뷰가 특히 낮은 것은 위에서 오는 정반사로 평균 채도가 27.9 까지 떨어져
# 문턱 35 를 밑돌기 때문이다.
#
# 표면 커버리지는 이제 동시 관측이 아니라 통과 중 회전에서 나온다. 세 면을
# 동시에 봐도 한 순간에 덮이는 것은 표면의 36.2% 뿐이고(세 대가 모두 위쪽을
# 보아 크게 겹친다), 탑뷰 한 대만으로도 순간을 여럿 모으면 그 이상을 덮는다.
DEFAULT_MIN_VIEWS_IN_ROI = 1

# 한 순간을 대표 프레임으로 채택하는 데 필요한 최소 면 수. 예전에는 카메라
# 수가 코드에 박혀 있어 파라미터로 낮출 수 없었다. 면이 더 보이면 그만큼
# 좋지만, 요구 조건이 되면 검사가 열리지 않는다.
DEFAULT_MIN_VIEWS_PER_INSTANT = 1
# 세 카메라는 같은 OnPlaybackTick에서 발행되지만 도착 순서와 stamp가 완전히
# 같다고 보장하지 않는다. docs/architecture/ros2_interfaces.md의 계약에 따라
# 한 검사의 세 view는 timestamp 최댓값과 최솟값 차이가 20ms 이내여야 한다.
GROUP_STAMP_TOLERANCE_NS = 20_000_000
DEFAULT_GROUP_QUEUE_SIZE = 8

# 검사에 필요한 것은 사과 주변뿐이다. 1280x720 전체를 보내면 view 한 장이
# 350KB를 넘어 UDP 조각으로 쪼개지고, 원본 카메라 스트림이 대역폭을 채운
# 상태에서 조각이 유실되어 검사 전체가 TIMEOUT 된다. 사과 bounding box로
# 잘라 보내면 데이터량이 크게 줄어 한 조각에 들어간다.
#
# 크롭 후 실측은 프레임당 84KB(최악 115KB)이고 그중 96.8%가 RGB다. 순간 8개를
# 모아도 한 검사가 2MB 남짓인데, 이 경로는 어댑터와 검사 노드가 같은 PC 에
# 있어 네트워크를 건너지 않는다. 실측 495~874 MB/s 로 24프레임이 3~4ms 에
# 도착한다. 그래서 순간 수를 늘리려고 RGB 를 버릴 이유가 없다. RGB 를 남겨야
# 임계값 사후 조정, 오분류 추적, 학습 데이터 수집이 가능하다.
DEFAULT_CROP_MARGIN_PX = 48

# 착색률을 판정할 수 없는 표면. docs/architecture/ros2_interfaces.md 는
# ignore_mask 를 "반사, 과도한 음영, 경계 등 착색률 계산에서 제외할 영역"으로
# 규정한다. 정반사로 하얗게 뜬 곳과 색을 분간할 수 없이 어두운 곳이 여기 든다.
# 이 영역을 분모에 남겨두면 완전히 빨간 사과도 78% 로 측정된다. 실측에서는
# 깊은 그늘이 표면의 15% 를 차지했다.
IGNORE_SPECULAR_MIN_VALUE = 230
IGNORE_SPECULAR_MAX_SATURATION = 60
IGNORE_SHADOW_MAX_VALUE = 25

# 사양이 규정한 세 번째 항목인 "경계". 실루엣 가장자리 픽셀은 사과 표면이
# 아니라 어두운 배경과 섞인 값이라 R/G·R/B 비율이 오염된다. 실측에서 이 띠는
# 표면의 4~9% 인데 남은 실패 픽셀의 35~47% 가 여기 몰려 있었다. 침식은
# apple_mask 안쪽으로만 들어가므로 사과가 아닌 영역을 새로 포함하지 않는다.
IGNORE_BOUNDARY_ERODE_PX = 2


try:
    import rclpy
    from appleproj_interfaces.msg import InspectionCompleted, InspectionImage
    from builtin_interfaces.msg import Time
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    from std_msgs.msg import Header

    _ROS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    rclpy = None  # type: ignore[assignment]
    InspectionCompleted = None  # type: ignore[assignment,misc]
    Time = None  # type: ignore[assignment,misc]
    InspectionImage = None  # type: ignore[assignment,misc]
    CameraInfo = None  # type: ignore[assignment,misc]
    CompressedImage = None  # type: ignore[assignment,misc]
    Image = None  # type: ignore[assignment,misc]
    Header = None  # type: ignore[assignment,misc]
    Node = object  # type: ignore[assignment,misc]
    Parameter = None  # type: ignore[assignment,misc]
    QoSProfile = None  # type: ignore[assignment,misc]
    ReliabilityPolicy = None  # type: ignore[assignment,misc]
    DurabilityPolicy = None  # type: ignore[assignment,misc]
    HistoryPolicy = None  # type: ignore[assignment,misc]
    _ROS_IMPORT_ERROR = exc


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass
class _PendingFrame:
    rgb: Any | None = None
    depth: Any | None = None
    camera_info: Any | None = None

    @property
    def complete(self) -> bool:
        return self.rgb is not None and self.depth is not None and self.camera_info is not None


@dataclass
class ExactStampSynchronizer:
    """Bounded exact-stamp synchronizer independent of ROS message_filters."""

    queue_size: int = DEFAULT_SYNC_QUEUE_SIZE
    _pending: OrderedDict[int, _PendingFrame] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")

    def add(self, component: str, message: Any) -> tuple[Any, Any, Any] | None:
        if component not in {"rgb", "depth", "camera_info"}:
            raise ValueError(f"unsupported synchronized component: {component}")
        key = stamp_to_ns(message.header.stamp)
        pending = self._pending.setdefault(key, _PendingFrame())
        setattr(pending, component, message)
        self._pending.move_to_end(key)
        while len(self._pending) > self.queue_size:
            self._pending.popitem(last=False)
        if not pending.complete:
            return None
        self._pending.pop(key, None)
        return pending.rgb, pending.depth, pending.camera_info


def _packed_rows(message: Any, bytes_per_pixel: int) -> bytes:
    import numpy as np

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    row_size = width * bytes_per_pixel
    if height <= 0 or width <= 0 or step < row_size:
        raise ValueError("raw image dimensions or step are invalid")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("raw image data is shorter than height * step")
    rows = raw[: height * step].reshape(height, step)
    return np.ascontiguousarray(rows[:, :row_size]).tobytes()


def decode_rgb_bgr(message: Any):
    """Decode common ROS Image RGB encodings into an OpenCV BGR array."""

    import cv2
    import numpy as np

    encoding = str(message.encoding).lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported RGB encoding: {message.encoding!r}")
    packed = _packed_rows(message, channels)
    image = np.frombuffer(packed, dtype=np.uint8).reshape(
        int(message.height),
        int(message.width),
        channels,
    )
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def decode_depth_mm(message: Any):
    """Decode 16-bit millimetres or 32-bit metres into uint16 millimetres."""

    import numpy as np

    encoding = str(message.encoding).lower()
    is_bigendian = bool(message.is_bigendian)
    if encoding in {"16uc1", "mono16"}:
        packed = _packed_rows(message, 2)
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        return np.frombuffer(packed, dtype=dtype).reshape(
            int(message.height), int(message.width)
        ).astype(np.uint16)
    if encoding == "32fc1":
        packed = _packed_rows(message, 4)
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
        depth_m = np.frombuffer(packed, dtype=dtype).reshape(
            int(message.height), int(message.width)
        )
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= 65.535)
        depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
        return depth_mm
    raise ValueError(f"unsupported depth encoding: {message.encoding!r}")


def selected_apple_mask(image, config: DetectionConfig = DetectionConfig()):
    """Return only the selected apple contour, excluding other saturated objects."""

    import cv2
    import numpy as np

    detection = detect_single_apple(image, config)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [detection.contour], -1, 255, thickness=cv2.FILLED)
    return detection, mask


def rolling_apple_mask(image, previous_detection, config: DetectionConfig):
    """Re-detect a rotating apple near its previous bounding box.

    The relaxed colour/shape thresholds are only applied inside an expanded
    previous box. This keeps bright or yellow-red peel visible while avoiding
    a full-frame relaxed search that could select conveyor or background parts.
    """

    import numpy as np

    height, width = image.shape[:2]
    x, y, box_width, box_height = previous_detection.bounding_box
    padding = int(
        round(max(box_width, box_height) * ROLLING_FALLBACK_BBOX_EXPANSION)
    )
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + box_width + padding)
    y1 = min(height, y + box_height + padding)
    if x1 <= x0 or y1 <= y0:
        raise AppleNotDetected("previous apple bounding box is outside the image")

    relaxed = DetectionConfig(
        min_saturation=min(
            config.min_saturation, ROLLING_FALLBACK_MIN_SATURATION
        ),
        min_value=max(10, config.min_value - 15),
        max_value=255,
        min_area_ratio=max(0.0001, config.min_area_ratio * 0.25),
        morphology_kernel=config.morphology_kernel,
        hue_low_max=max(config.hue_low_max, ROLLING_FALLBACK_HUE_LOW_MAX),
        hue_high_min=config.hue_high_min,
        min_fill_ratio=max(0.35, config.min_fill_ratio - 0.20),
        max_area_ratio=config.max_area_ratio,
    )
    local_detection, local_mask = selected_apple_mask(
        image[y0:y1, x0:x1], relaxed
    )
    previous_diameter = max(float(previous_detection.diameter_px), 1.0)
    diameter_ratio = float(local_detection.diameter_px) / previous_diameter
    if not (
        ROLLING_FALLBACK_MIN_DIAMETER_RATIO
        <= diameter_ratio
        <= ROLLING_FALLBACK_MAX_DIAMETER_RATIO
    ):
        raise AppleNotDetected(
            "rolling fallback candidate changed diameter too much: "
            f"ratio={diameter_ratio:.2f}"
        )

    contour = np.asarray(local_detection.contour).copy()
    contour[..., 0] += x0
    contour[..., 1] += y0
    local_x, local_y, local_width, local_height = local_detection.bounding_box
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = local_mask
    detection = replace(
        local_detection,
        contour=contour,
        bounding_box=(
            local_x + x0,
            local_y + y0,
            local_width,
            local_height,
        ),
        center=(
            local_detection.center[0] + x0,
            local_detection.center[1] + y0,
        ),
        mask=full_mask,
    )
    return detection, full_mask


def encode_image(extension: str, image) -> bytes:
    import cv2

    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"OpenCV failed to encode {extension} image")
    return encoded.tobytes()


def annotate_apple_detection(image, detection=None):
    """Draw the selected apple bounding box without modifying the source image."""

    import cv2

    annotated = image.copy()
    if detection is None:
        cv2.putText(
            annotated,
            "NO APPLE",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    x, y, width, height = detection.bounding_box
    cv2.rectangle(
        annotated,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 0),
        2,
    )
    label_y = max(20, y - 8)
    cv2.putText(
        annotated,
        f"apple {detection.confidence:.2f} ({x},{y},{width},{height})",
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def resize_debug_image(image, scale: float):
    """Downscale display-only output while preserving inspection resolution."""

    import cv2

    if not 0.0 < scale <= 1.0:
        raise ValueError("detection debug scale must satisfy 0 < scale <= 1")
    if scale == 1.0:
        return image.copy()
    height, width = image.shape[:2]
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def encode_debug_jpeg(image, quality: int) -> bytes:
    """Encode display-only BGR output as a bounded-quality JPEG."""

    import cv2

    if not 1 <= quality <= 100:
        raise ValueError("detection debug JPEG quality must be between 1 and 100")
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not success:
        raise ValueError("OpenCV failed to encode detection debug JPEG")
    return encoded.tobytes()


def make_reliable_qos(depth: int = 10) -> Any:
    if QoSProfile is None:
        raise RuntimeError("rclpy is required to construct QoS profiles")
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def unmeasurable_surface_mask(image_bgr, apple_mask):
    """Apple pixels whose colour cannot be judged: highlight, deep shadow, edge.

    A specular highlight loses its hue, and a pixel dark enough to be near black
    carries no usable colour either. The silhouette band is a third case: those
    pixels blend the apple with whatever is behind it, so their channel ratios
    describe the mix rather than the skin. All three belong in ignore_mask so the
    colour ratio divides by the surface it could actually measure.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    specular = (value >= IGNORE_SPECULAR_MIN_VALUE) & (
        saturation <= IGNORE_SPECULAR_MAX_SATURATION
    )
    shadow = value <= IGNORE_SHADOW_MAX_VALUE

    inside = apple_mask > 0
    if IGNORE_BOUNDARY_ERODE_PX > 0:
        # iterations 는 키워드로 넘긴다. cv2.erode 의 세 번째 위치 인자는
        # dst 이므로 위치로 넘기면 침식이 1회로 조용히 고정된다.
        eroded = cv2.erode(
            np.asarray(inside, dtype=np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=IGNORE_BOUNDARY_ERODE_PX,
        ) > 0
    else:
        eroded = inside
    boundary = inside & ~eroded

    return np.asarray(
        (specular | shadow | boundary) & inside, dtype=np.uint8
    ) * 255


# 순간 수에 대한 표면 커버리지 추정. inspection_session.py 에 기록된 실측값
# (순간 1개 36.2%, 4개 87.0%, 8개 99.6%)을 앵커로 선형 보간한다. 이 값들은 한
# 순간에 3면이 모두 잡힌 경우를 전제하므로, 면이 빠진 순간이 섞이면 실제
# 커버리지는 이보다 낮다. 로그에 상황을 드러내기 위한 근사치이지 등급 판정에
# 쓰는 값이 아니다.
_COVERAGE_ANCHORS = ((0, 0.0), (1, 0.362), (4, 0.870), (8, 0.996))


def estimated_surface_coverage(instant_count: int) -> float:
    """Rough share of the peel covered by this many three-view instants."""
    if instant_count <= 0:
        return 0.0
    last_count, last_value = _COVERAGE_ANCHORS[-1]
    if instant_count >= last_count:
        return last_value
    for (x0, y0), (x1, y1) in zip(_COVERAGE_ANCHORS, _COVERAGE_ANCHORS[1:]):
        if x0 <= instant_count <= x1:
            return y0 + (y1 - y0) * (instant_count - x0) / (x1 - x0)
    return last_value


def apple_position_m(apple_mask, depth_mm, camera_info):
    """Back-project the apple centroid into the camera optical frame, in metres.

    Returns ``None`` when the apple carries no valid depth, which happens on a
    specular frame or when the depth stream lags the colour one.

    The median depth over the mask is used rather than the depth at the centroid
    pixel: a single pixel on a glossy apple is frequently a dropout.
    """
    import numpy as np

    inside = apple_mask > 0
    if not bool(inside.any()):
        return None
    depth = np.asarray(depth_mm)
    valid = inside & (depth > 0)
    if not bool(valid.any()):
        return None
    z_m = float(np.median(depth[valid])) / 1000.0
    if z_m <= 0.0:
        return None

    rows, columns = np.nonzero(inside)
    u = float(columns.mean())
    v = float(rows.mean())
    k = [float(value) for value in camera_info.k]
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    if fx == 0.0 or fy == 0.0:
        return None
    return ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)


def apple_crop_box(apple_mask, margin_px: int) -> tuple[int, int, int, int]:
    """Bounding box of the apple mask, grown by margin and clamped to frame."""
    import numpy as np

    rows = np.nonzero(apple_mask.any(axis=1))[0]
    columns = np.nonzero(apple_mask.any(axis=0))[0]
    if rows.size == 0 or columns.size == 0:
        raise ValueError("apple mask is empty; nothing to crop")
    height, width = apple_mask.shape[:2]
    x0 = max(0, int(columns[0]) - margin_px)
    y0 = max(0, int(rows[0]) - margin_px)
    x1 = min(width, int(columns[-1]) + 1 + margin_px)
    y1 = min(height, int(rows[-1]) + 1 + margin_px)
    return x0, y0, x1, y1


def cropped_camera_info(camera_info: Any, box: tuple[int, int, int, int]) -> Any:
    """Shift the principal point so the crop keeps a valid projection model.

    depth_geometry back-projects with (x - cx) * z / fx on image pixels, so a
    cropped image needs cx and cy moved by the crop origin. roi records the
    original window for traceability.
    """
    import numpy as np

    x0, y0, x1, y1 = box
    info = copy.deepcopy(camera_info)
    info.width = int(x1 - x0)
    info.height = int(y1 - y0)

    # CameraInfo.k/.p are fixed-size float64 arrays in rclpy. Rebuilding them
    # through type(...) would treat the values as a shape, so write the values
    # back in place instead.
    k = [float(value) for value in info.k]
    k[2] -= x0
    k[5] -= y0
    p = [float(value) for value in info.p]
    p[2] -= x0
    p[6] -= y0
    if isinstance(info.k, np.ndarray):
        info.k[:] = k
    else:
        info.k = k
    if isinstance(info.p, np.ndarray):
        info.p[:] = p
    else:
        info.p = p

    info.roi.x_offset = int(x0)
    info.roi.y_offset = int(y0)
    info.roi.width = int(x1 - x0)
    info.roi.height = int(y1 - y0)
    info.roi.do_rectify = False
    return info


def make_sensor_qos(reliable: bool, depth: int = 10) -> Any:
    """QoS for the raw camera streams.

    docs/architecture/ros2_interfaces.md puts sensor streams on Sensor Data
    QoS. Subscribing BEST_EFFORT also stays compatible with a RELIABLE
    publisher, while a RELIABLE subscriber silently receives nothing from a
    BEST_EFFORT one.
    """
    if QoSProfile is None:
        raise RuntimeError("rclpy is required to construct QoS profiles")
    return QoSProfile(
        reliability=(
            ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class ConveyorCameraAdapterNode(Node):  # type: ignore[misc]
    """Group the three fixed conveyor views of one apple into one inspection.

    The cameras publish from a shared OnPlaybackTick, so all three views of a
    tick carry the same stamp.  Frames are grouped by stamp and emitted as one
    inspection whose frame_index enumerates the views; header.frame_id keeps
    each camera's own optical frame so the grader knows which face it sees.
    """

    def __init__(self) -> None:
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable; source ROS 2 Jazzy and "
                "the built workspace first"
            ) from _ROS_IMPORT_ERROR
        super().__init__(
            "conveyor_camera_adapter_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        for name, default in (
            ("camera_namespaces", list(DEFAULT_CAMERA_NAMESPACES)),
            ("roi_mode", DEFAULT_ROI_MODE),
            ("roi_edge_margin_px", DEFAULT_ROI_EDGE_MARGIN_PX),
            ("roi_min_x_ratio", DEFAULT_ROI_MIN_X_RATIO),
            ("roi_max_x_ratio", DEFAULT_ROI_MAX_X_RATIO),
            ("roi_exit_patience", DEFAULT_ROI_EXIT_PATIENCE),
            ("roi_entry_patience", DEFAULT_ROI_ENTRY_PATIENCE),
            ("min_views_in_roi", DEFAULT_MIN_VIEWS_IN_ROI),
            ("min_views_per_instant", DEFAULT_MIN_VIEWS_PER_INSTANT),
            ("crop_margin_px", DEFAULT_CROP_MARGIN_PX),
            ("camera_qos_reliable", False),
            ("rearm_absent_frames", DEFAULT_REARM_ABSENT_FRAMES),
            ("min_saturation", DEFAULT_ROLLING_MIN_SATURATION),
            ("min_value", DetectionConfig().min_value),
            ("max_value", DEFAULT_ROLLING_MAX_VALUE),
            ("min_area_ratio", DetectionConfig().min_area_ratio),
            ("morphology_kernel", DetectionConfig().morphology_kernel),
            ("representative_instants", REPRESENTATIVE_INSTANTS),
            ("min_instant_gap_ns", MIN_INSTANT_GAP_NS),
            ("roi_aim_distance_m", DEFAULT_ROI_AIM_DISTANCE_M),
            ("roi_radius_m", DEFAULT_ROI_RADIUS_M),
            ("track_gap_ns", DEFAULT_TRACK_GAP_NS),
            ("track_radius_m", DEFAULT_TRACK_RADIUS_M),
            ("detection_debug_topic", DETECTION_DEBUG_TOPIC),
            ("detection_debug_scale", DEFAULT_DETECTION_DEBUG_SCALE),
            (
                "detection_debug_jpeg_quality",
                DEFAULT_DETECTION_DEBUG_JPEG_QUALITY,
            ),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        self._namespaces = [
            str(value) for value in self.get_parameter("camera_namespaces").value
        ]
        if not self._namespaces:
            raise ValueError("camera_namespaces must not be empty")
        self._rearm_absent_frames = int(
            self.get_parameter("rearm_absent_frames").value
        )
        if self._rearm_absent_frames < 1:
            raise ValueError("rearm_absent_frames must be positive")
        self._roi_min_x_ratio = float(self.get_parameter("roi_min_x_ratio").value)
        self._roi_max_x_ratio = float(self.get_parameter("roi_max_x_ratio").value)
        if not 0.0 <= self._roi_min_x_ratio < self._roi_max_x_ratio <= 1.0:
            raise ValueError(
                "roi ratios must satisfy 0 <= min < max <= 1"
            )
        self._roi_mode = str(self.get_parameter("roi_mode").value)
        if self._roi_mode not in (ROI_MODE_AIM_SPHERE, ROI_MODE_FULL_FRAME):
            raise ValueError(
                f"roi_mode must be {ROI_MODE_AIM_SPHERE!r} or "
                f"{ROI_MODE_FULL_FRAME!r}"
            )
        self._roi_edge_margin_px = int(
            self.get_parameter("roi_edge_margin_px").value
        )
        if self._roi_edge_margin_px < 0:
            raise ValueError("roi_edge_margin_px must be non-negative")
        self._roi_exit_patience = int(self.get_parameter("roi_exit_patience").value)
        if self._roi_exit_patience < 1:
            raise ValueError("roi_exit_patience must be positive")
        self._roi_entry_patience = int(self.get_parameter("roi_entry_patience").value)
        if self._roi_entry_patience < 1:
            raise ValueError("roi_entry_patience must be positive")
        configured_min_views = int(self.get_parameter("min_views_in_roi").value)
        # 0 이면 카메라 수를 그대로 요구한다.
        self._min_views_in_roi = configured_min_views or len(self._namespaces)
        if not 1 <= self._min_views_in_roi <= len(self._namespaces):
            raise ValueError(
                "min_views_in_roi must be between 1 and the number of cameras"
            )
        configured_per_instant = int(
            self.get_parameter("min_views_per_instant").value
        )
        self._min_views_per_instant = (
            configured_per_instant or len(self._namespaces)
        )
        if not 1 <= self._min_views_per_instant <= len(self._namespaces):
            raise ValueError(
                "min_views_per_instant must be between 1 and the number of cameras"
            )
        self._crop_margin_px = int(self.get_parameter("crop_margin_px").value)
        if self._crop_margin_px < 0:
            raise ValueError("crop_margin_px must be non-negative")
        self._representative_instants = int(
            self.get_parameter("representative_instants").value
        )
        if self._representative_instants < 1:
            raise ValueError("representative_instants must be at least 1")
        self._min_instant_gap_ns = int(
            self.get_parameter("min_instant_gap_ns").value
        )
        if self._min_instant_gap_ns < 0:
            raise ValueError("min_instant_gap_ns must be non-negative")
        self._roi_aim_distance_m = float(
            self.get_parameter("roi_aim_distance_m").value
        )
        self._roi_radius_m = float(self.get_parameter("roi_radius_m").value)
        if self._roi_aim_distance_m <= 0.0 or self._roi_radius_m <= 0.0:
            raise ValueError("roi_aim_distance_m and roi_radius_m must be positive")
        self._track_gap_ns = int(self.get_parameter("track_gap_ns").value)
        self._track_radius_m = float(self.get_parameter("track_radius_m").value)
        self._last_track: dict[str, Any] | None = None
        self._detection_debug_scale = float(
            self.get_parameter("detection_debug_scale").value
        )
        if not 0.0 < self._detection_debug_scale <= 1.0:
            raise ValueError("detection_debug_scale must satisfy 0 < scale <= 1")
        self._detection_debug_jpeg_quality = int(
            self.get_parameter("detection_debug_jpeg_quality").value
        )
        if not 1 <= self._detection_debug_jpeg_quality <= 100:
            raise ValueError(
                "detection_debug_jpeg_quality must be between 1 and 100"
            )
        if self._representative_instants * len(self._namespaces) > MAX_REPRESENTATIVE_FRAMES:
            raise ValueError(
                "representative_instants x cameras cannot exceed "
                f"MAX_REPRESENTATIVE_FRAMES ({MAX_REPRESENTATIVE_FRAMES})"
            )
        self._detection_config = DetectionConfig(
            min_saturation=int(self.get_parameter("min_saturation").value),
            min_value=int(self.get_parameter("min_value").value),
            max_value=int(self.get_parameter("max_value").value),
            min_area_ratio=float(self.get_parameter("min_area_ratio").value),
            morphology_kernel=int(self.get_parameter("morphology_kernel").value),
        )
        self._synchronizers = {
            namespace: ExactStampSynchronizer() for namespace in self._namespaces
        }
        # 카메라별 최근 view 버퍼. stamp가 정확히 같지 않아도 되므로 stamp를
        # 키로 쓰지 않고 카메라마다 후보를 모아 tolerance 안에서 짝을 찾는다.
        self._groups: dict[str, list[tuple[int, tuple[Any, Any, Any]]]] = {
            namespace: [] for namespace in self._namespaces
        }
        # 한 사과의 ROI 진입부터 이탈까지가 하나의 검사 세션이다.
        self._session: dict[str, Any] | None = None
        self._outside_groups = 0
        self._inside_groups = 0
        self._inspection_sequence = 0
        # ROI 안에 사과가 없으면 아무 로그도 남지 않아 "영상이 안 온다"와
        # "사과가 없다"를 구분할 수 없다. 주기 상태 로그로 그 둘을 나눈다.
        self._counters = {
            "views": 0,
            "groups": 0,
            "detected": 0,
            "in_roi": 0,
            "ambiguous": 0,
            # 완료된 검사 수. 주기당 증가량이 곧 사과 처리율이고, 여기에
            # 통과 시간을 곱하면 시야 안에 사과가 몇 개나 함께 있는지 나온다.
            # full_frame 으로 넓힐 때 다중 사과 추적이 필요한지의 근거가 된다.
            "sessions": 0,
        }
        # 마지막 그룹의 검출 위치·크기. 정지한 배경을 사과로 오인하는지
        # 판단하려면 무엇이 어디서 잡히는지 보여야 한다.
        self._last_detection: list[str] = []
        # RGB 도착 즉시 검출해 debug 영상을 발행한다. 이후 같은 RGB-D 그룹은
        # 이 결과를 재사용하여 OpenCV 검출을 두 번 수행하지 않는다.
        self._rgb_detection_cache: OrderedDict[
            tuple[str, int], tuple[Any, Any, Any] | None
        ] = OrderedDict()
        self._rolling_detections: dict[str, Any] = {}
        self._rolling_detection_misses: dict[str, int] = {
            namespace: 0 for namespace in self._namespaces
        }

        qos = make_reliable_qos()
        self._inspection_publisher = self.create_publisher(
            InspectionImage, INSPECTION_TOPIC, qos
        )
        self._completion_publisher = self.create_publisher(
            InspectionCompleted, COMPLETION_TOPIC, qos
        )
        detection_debug_topic = str(
            self.get_parameter("detection_debug_topic").value
        )
        self._detection_debug_publisher = self.create_publisher(
            CompressedImage,
            detection_debug_topic,
            make_sensor_qos(False, depth=1),
        )
        camera_reliable = bool(self.get_parameter("camera_qos_reliable").value)
        camera_qos = make_sensor_qos(camera_reliable)
        for namespace in self._namespaces:
            self._subscribe_camera(namespace, camera_qos)

        self.get_logger().info(
            "GPU PC 2 conveyor adapter ready: "
            f"{len(self._namespaces)} views {self._namespaces} -> "
            f"{INSPECTION_TOPIC}; bbox debug -> {detection_debug_topic} "
            f"(JPEG q={self._detection_debug_jpeg_quality}, "
            f"scale {self._detection_debug_scale:.2f}, latest frame only)"
        )
        self.get_logger().info(
            f"ROI mode: {self._roi_mode} "
            + (
                f"(edge margin {self._roi_edge_margin_px} px)"
                if self._roi_mode == ROI_MODE_FULL_FRAME
                else f"(aim {self._roi_aim_distance_m:.2f} m, "
                f"radius {self._roi_radius_m:.2f} m; depth-less fallback band "
                f"x {self._roi_min_x_ratio:.2f}-{self._roi_max_x_ratio:.2f})"
            )
            + f", exit patience {self._roi_exit_patience} groups"
        )
        self.get_logger().info(
            "camera QoS: "
            f"{'RELIABLE' if camera_reliable else 'BEST_EFFORT'} / VOLATILE "
            "(a RELIABLE subscriber receives nothing from a BEST_EFFORT publisher)"
        )
        self.create_timer(5.0, self._log_status)

    def _log_status(self) -> None:
        counters = self._counters
        state = "idle" if self._session is None else self._session["inspection_id"]
        self.get_logger().info(
            f"status: views={counters['views']} groups={counters['groups']} "
            f"apple_detected={counters['detected']} in_roi={counters['in_roi']} "
            f"multi_apple={counters['ambiguous']} "
            f"inspections={counters['sessions']} "
            f"session={state}"
        )
        if self._last_detection:
            self.get_logger().info(
                "  last detection  " + " | ".join(self._last_detection)
            )
        if counters["groups"] and counters["detected"] == counters["groups"]:
            self.get_logger().warn(
                "an apple is detected in every single group; a moving apple "
                "should be absent part of the time, so this is likely a static "
                "false positive on scene colour"
            )
        if counters["views"] == 0:
            self.get_logger().warn(
                "no camera frames received; check topic names, QoS and "
                "ROS_DOMAIN_ID against the publisher"
            )
        elif counters["groups"] == 0:
            self.get_logger().warn(
                "frames arrive but never group; the three views are further "
                "apart than the 20 ms contract window"
            )
        elif counters["detected"] == 0:
            self.get_logger().warn(
                "views group but no apple is detected; check lighting and the "
                "HSV detection parameters"
            )
        elif counters["in_roi"] == 0:
            self.get_logger().warn(
                "apples are detected but never inside the ROI band; adjust "
                "roi_min_x_ratio / roi_max_x_ratio"
            )

    def _subscribe_camera(self, namespace: str, qos: Any) -> None:
        for component, suffix, message_type in (
            ("rgb", RGB_SUFFIX, Image),
            ("depth", DEPTH_SUFFIX, Image),
            ("camera_info", CAMERA_INFO_SUFFIX, CameraInfo),
        ):
            self.create_subscription(
                message_type,
                f"{namespace}{suffix}",
                # namespace/component are bound per subscription, not captured late.
                lambda message, ns=namespace, part=component: self._accept(
                    ns, part, message
                ),
                qos,
            )

    def _accept(self, namespace: str, component: str, message: Any) -> None:
        if component == "rgb":
            self._detect_and_publish_rgb(namespace, message)
        synchronized = self._synchronizers[namespace].add(component, message)
        if synchronized is None:
            return
        self._counters["views"] += 1
        try:
            self._validate_view(*synchronized)
        except Exception as exc:
            self.get_logger().error(
                f"Rejected {namespace} RGB-D frame: {type(exc).__name__}: {exc}"
            )
            return
        self._collect(namespace, synchronized)

    def _detect_and_publish_rgb(self, namespace: str, rgb_message: Any) -> None:
        """Detect and publish a bounding box from RGB without waiting for depth."""

        key = (namespace, stamp_to_ns(rgb_message.header.stamp))
        try:
            image = decode_rgb_bgr(rgb_message)
            try:
                detection, apple_mask = self._detect_rolling_apple(namespace, image)
                cached: tuple[Any, Any, Any] | None = (
                    image,
                    detection,
                    apple_mask,
                )
                annotated = annotate_apple_detection(image, detection)
            except AppleNotDetected:
                self._record_rolling_detection_miss(namespace)
                cached = None
                annotated = annotate_apple_detection(image)

            self._rgb_detection_cache[key] = cached
            self._rgb_detection_cache.move_to_end(key)
            max_cached = DEFAULT_SYNC_QUEUE_SIZE * max(1, len(self._namespaces))
            while len(self._rgb_detection_cache) > max_cached:
                self._rgb_detection_cache.popitem(last=False)

            debug_image = resize_debug_image(
                annotated, self._detection_debug_scale
            )
            debug_message = CompressedImage()
            debug_message.header = copy.deepcopy(rgb_message.header)
            debug_message.format = "bgr8; jpeg compressed bgr8"
            debug_message.data = encode_debug_jpeg(
                debug_image, self._detection_debug_jpeg_quality
            )
            self._detection_debug_publisher.publish(debug_message)
        except Exception as exc:
            self.get_logger().error(
                f"Failed to detect {namespace} RGB frame: "
                f"{type(exc).__name__}: {exc}"
            )

    def _detect_rolling_apple(self, namespace: str, image):
        """Use normal detection first, then a previous-box rolling fallback."""

        try:
            detection, apple_mask = selected_apple_mask(
                image, self._detection_config
            )
        except AppleNotDetected:
            previous = getattr(self, "_rolling_detections", {}).get(namespace)
            if previous is None:
                raise
            detection, apple_mask = rolling_apple_mask(
                image, previous, self._detection_config
            )
        if not hasattr(self, "_rolling_detections"):
            self._rolling_detections = {}
        if not hasattr(self, "_rolling_detection_misses"):
            self._rolling_detection_misses = {}
        self._rolling_detections[namespace] = detection
        self._rolling_detection_misses[namespace] = 0
        return detection, apple_mask

    def _record_rolling_detection_miss(self, namespace: str) -> None:
        if not hasattr(self, "_rolling_detection_misses"):
            self._rolling_detection_misses = {}
        if not hasattr(self, "_rolling_detections"):
            self._rolling_detections = {}
        misses = getattr(self, "_rolling_detection_misses", {}).get(namespace, 0) + 1
        self._rolling_detection_misses[namespace] = misses
        if misses >= getattr(
            self, "_rearm_absent_frames", DEFAULT_REARM_ABSENT_FRAMES
        ):
            self._rolling_detections.pop(namespace, None)

    @staticmethod
    def _validate_view(rgb_message: Any, depth_message: Any, camera_info: Any) -> None:
        signatures = {
            (
                stamp_to_ns(message.header.stamp),
                str(message.header.frame_id),
            )
            for message in (rgb_message, depth_message, camera_info)
        }
        if len(signatures) != 1:
            raise ValueError("RGB, depth and CameraInfo stamps/frame_ids must match")
        frame_id = str(rgb_message.header.frame_id)
        if frame_id not in QUALITY_CAMERA_OPTICAL_FRAMES:
            raise ValueError(
                f"frame_id {frame_id!r} is not a known conveyor camera frame"
            )
        if (
            int(rgb_message.width) != int(depth_message.width)
            or int(rgb_message.height) != int(depth_message.height)
            or int(rgb_message.width) != int(camera_info.width)
            or int(rgb_message.height) != int(camera_info.height)
        ):
            raise ValueError("RGB, depth and CameraInfo dimensions must match")

    def _collect(self, namespace: str, view: tuple[Any, Any, Any]) -> None:
        """Buffer each view until all three cameras land within tolerance."""
        stamp_ns = stamp_to_ns(view[0].header.stamp)
        pending = self._groups.setdefault(namespace, [])
        pending.append((stamp_ns, view))
        while len(pending) > DEFAULT_GROUP_QUEUE_SIZE:
            pending.pop(0)

        group = self._match_group(stamp_ns)
        if group is None:
            return
        for name, (matched_stamp, _matched_view) in group.items():
            self._groups[name] = [
                entry for entry in self._groups[name] if entry[0] > matched_stamp
            ]
        stamps = [stamp for stamp, _ in group.values()]
        try:
            self._process_group(
                max(stamps),
                {name: entry[1] for name, entry in group.items()},
            )
        except Exception as exc:
            self.get_logger().error(
                f"Rejected conveyor view group: {type(exc).__name__}: {exc}"
            )

    def _match_group(self, reference_ns: int):
        """Pick one view per camera whose stamps span at most the tolerance."""
        group: dict[str, tuple[int, tuple[Any, Any, Any]]] = {}
        for namespace in self._namespaces:
            candidates = self._groups.get(namespace) or []
            if not candidates:
                return None
            nearest = min(
                candidates, key=lambda entry: abs(entry[0] - reference_ns)
            )
            if abs(nearest[0] - reference_ns) > GROUP_STAMP_TOLERANCE_NS:
                return None
            group[namespace] = nearest
        stamps = [stamp for stamp, _ in group.values()]
        if max(stamps) - min(stamps) > GROUP_STAMP_TOLERANCE_NS:
            return None
        return group

    def _process_group(
        self, stamp_ns: int, group: dict[str, tuple[Any, Any, Any]]
    ) -> None:
        self._counters["groups"] += 1
        views = []
        # frame_index는 카메라 위치로 고정한다: 위 0, 왼쪽 1, 오른쪽 2.
        # 검출 순서로 매기면 한 면이 빠졌을 때 index가 다른 면을 가리킨다.
        for frame_index, namespace in enumerate(self._namespaces):
            rgb_message, depth_message, camera_info = group[namespace]
            cache_key = (namespace, stamp_to_ns(rgb_message.header.stamp))
            cache_missing = object()
            cached = getattr(self, "_rgb_detection_cache", {}).pop(
                cache_key, cache_missing
            )
            if cached is cache_missing:
                image = decode_rgb_bgr(rgb_message)
                try:
                    detection, apple_mask = self._detect_rolling_apple(
                        namespace, image
                    )
                except AppleNotDetected:
                    self._record_rolling_detection_miss(namespace)
                    # 세 면 중 일부에만 사과가 보이는 것은 정상이다.
                    continue
            elif cached is None:
                continue
            else:
                image, detection, apple_mask = cached
            if detection.candidate_count > 1:
                # docs/features/conveyor.md: 두 사과가 겹치면 RECHECK 대상이다.
                # 붙어 있는 두 사과는 한 윤곽으로 합쳐져 직경이 두 배가 되므로
                # 이 면은 대표 프레임 후보에서 제외한다.
                self._counters["ambiguous"] += 1
                continue
            views.append(
                {
                    "frame_index": frame_index,
                    "rgb_message": rgb_message,
                    "camera_info": camera_info,
                    "image": image,
                    "apple_mask": apple_mask,
                    "depth_mm": decode_depth_mm(depth_message),
                    "bounding_box": detection.bounding_box,
                    "confidence": detection.confidence,
                    "diameter_px": detection.diameter_px,
                    "position_m": apple_position_m(
                        apple_mask, decode_depth_mm(depth_message), camera_info
                    ),
                    "in_roi": self._view_in_roi(
                        apple_mask, decode_depth_mm(depth_message), camera_info
                    ),
                }
            )

        if views:
            self._counters["detected"] += 1
            import numpy as np

            self._last_detection = [
                "{}: bbox={} x={:.2f} area={}px d={:.0f}px".format(
                    ("top", "left", "right")[view["frame_index"]]
                    if view["frame_index"] < 3
                    else view["frame_index"],
                    view["bounding_box"],
                    self._mask_centre_ratio(view["apple_mask"]) or -1.0,
                    int(np.count_nonzero(view["apple_mask"])),
                    view["diameter_px"],
                )
                for view in views
            ]
        inside = [view for view in views if view["in_roi"]]
        # 판정에 필요한 면 수만큼 동시에 ROI 안에 있어야 한 검사로 인정한다.
        if len(inside) < self._min_views_in_roi:
            inside = []
        if inside:
            self._counters["in_roi"] += 1
            self._outside_groups = 0
            self._inside_groups += 1
            if self._session is None:
                if self._inside_groups < self._roi_entry_patience:
                    # 아직 깜빡임과 구분되지 않는다. 세션을 열지 않는다.
                    return
                self._begin_session(stamp_ns, inside)
            # 계약상 한 검사는 3면 한 순간이다. 통과하는 동안 후보만 모아 두고
            # ROI 이탈 시 가장 좋은 순간 하나를 대표 프레임으로 발행한다.
            self._consider_candidate(stamp_ns, inside)
            return

        # ROI 밖이거나 세 면 모두에서 사과를 놓친 그룹이다.
        self._inside_groups = 0
        if self._session is None:
            return
        self._outside_groups += 1
        if self._outside_groups >= self._roi_exit_patience:
            self._finish_session(stamp_ns)

    def _mask_centre_ratio(self, apple_mask):
        """Horizontal centroid of the mask as a fraction of image width."""
        import numpy as np

        columns = np.nonzero(apple_mask.any(axis=0))[0]
        if columns.size == 0:
            return None
        return float(columns.mean()) / float(apple_mask.shape[1])

    def _mask_in_roi(self, apple_mask) -> bool:
        """Fallback ROI test on image x, used only when depth is unusable."""
        centre_ratio = self._mask_centre_ratio(apple_mask)
        if centre_ratio is None:
            return False
        return self._roi_min_x_ratio <= centre_ratio <= self._roi_max_x_ratio

    def _mask_clear_of_border(self, apple_mask) -> bool:
        """True when the whole silhouette is inside the frame, margin included.

        An apple cut by the image edge has a truncated silhouette, so its
        diameter and its colour ratio would both be measured against a surface
        that is not all there. In full_frame mode this is the only thing that
        bounds the inspection region, so it has to carry that check on its own.
        """
        import numpy as np

        rows = np.nonzero(apple_mask.any(axis=1))[0]
        columns = np.nonzero(apple_mask.any(axis=0))[0]
        if rows.size == 0 or columns.size == 0:
            return False
        height, width = apple_mask.shape[:2]
        margin = self._roi_edge_margin_px
        return (
            int(rows[0]) >= margin
            and int(columns[0]) >= margin
            and int(rows[-1]) < height - margin
            and int(columns[-1]) < width - margin
        )

    def _view_in_roi(self, apple_mask, depth_mm, camera_info) -> bool:
        """True when this view should contribute to the current inspection.

        In full_frame mode the whole strip of conveyor the camera sees is the
        inspection region, so every apple clear of the border counts.

        In aim_sphere mode the region is a tolerance sphere around the aim
        point. Every camera is aimed at the same inspection point, so in each
        camera's optical frame that point sits on the optical axis at the
        configured aim distance. Measuring the distance to it makes all three
        cameras test one physical question instead of three different
        image-space ones.
        """
        if self._roi_mode == ROI_MODE_FULL_FRAME:
            return self._mask_clear_of_border(apple_mask)
        position = apple_position_m(apple_mask, depth_mm, camera_info)
        if position is None:
            # Depth dropped out; fall back to the image-space band rather than
            # discarding the view outright.
            return self._mask_in_roi(apple_mask)
        x, y, z = position
        offset = (x * x + y * y + (z - self._roi_aim_distance_m) ** 2) ** 0.5
        return offset <= self._roi_radius_m

    @staticmethod
    def _master_position_m(views: list[dict[str, Any]]):
        """Position from the lowest-numbered camera that produced one.

        _group_position_m averages coordinates that belong to three different
        optical frames, so its value shifts whenever a view drops out. For
        measuring how far the apple actually travelled, take one camera and
        stay with it: the numbers then live in a single frame throughout.
        """
        for view in sorted(views, key=lambda item: item["frame_index"]):
            if view.get("position_m"):
                return view["position_m"]
        return None

    @staticmethod
    def _group_position_m(views: list[dict[str, Any]]):
        """Mean 3D apple position over the views that produced one."""
        points = [view["position_m"] for view in views if view.get("position_m")]
        if not points:
            return None
        count = float(len(points))
        return tuple(sum(p[axis] for p in points) / count for axis in range(3))

    def _resolve_apple_id(self, stamp_ns: int, position, fallback: str) -> str:
        """Reuse the previous apple_id when this is the same fruit coming back.

        Detection drops out at the edges of the ROI, which closes the session and
        opens a new one on the very same apple. Minting an id per session made
        one apple appear as several, and the spec calls for a tracker rather than
        a fresh id each time.
        """
        track = self._last_track
        if track is None or position is None or track.get("position") is None:
            return fallback
        if stamp_ns - int(track["stamp_ns"]) > self._track_gap_ns:
            return fallback
        previous = track["position"]
        moved = sum((a - b) ** 2 for a, b in zip(position, previous)) ** 0.5
        if moved > self._track_radius_m:
            return fallback
        self.get_logger().info(
            f"tracker: same apple as {track['apple_id']} "
            f"({moved * 100:.1f} cm from the last sighting, "
            f"{(stamp_ns - int(track['stamp_ns'])) / 1e9:.2f} s ago)"
        )
        return str(track["apple_id"])

    def _begin_session(self, stamp_ns: int, views: list[dict[str, Any]]) -> None:
        self._inspection_sequence += 1
        suffix = f"{stamp_ns}-{self._inspection_sequence}"
        position = self._group_position_m(views)
        apple_id = self._resolve_apple_id(stamp_ns, position, f"apple-{suffix}")
        self._session = {
            "inspection_id": f"inspection-{suffix}",
            "apple_id": apple_id,
            "entry_stamp_ns": stamp_ns,
            "frames_sent": 0,
            "instants": [],
            "candidates": 0,
            "last_position_m": position,
            # 진단용. 기준 카메라 한 대의 위치만 시간순으로 모아 두면 통과
            # 거리와 실제 속도를 잴 수 있다. 이 둘을 알아야 간격과 ROI 크기를
            # 추측이 아니라 계산으로 정할 수 있다.
            "path": [],
            # ROI 안에 동시에 몇 면이 잡혔는지의 분포. 3면 동시 조건이 세션을
            # 쪼개고 있는지 여기서 드러난다.
            "views_histogram": {},
        }
        self._counters["sessions"] += 1
        self.get_logger().info(
            f"ROI entry: {self._session['inspection_id']} at {stamp_ns} ns"
        )

    @staticmethod
    def _view_sharpness(image) -> float:
        """Laplacian variance; higher is sharper. Used to drop motion blur."""
        import cv2

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(grey, cv2.CV_64F).var())

    def _crop_view(self, view: dict[str, Any]) -> dict[str, Any] | None:
        """Reduce a view to just the apple, so a kept instant stays small.

        Cropping here rather than at publish time matters: a session holds
        several instants at once, and a full 1280x720 view is about 5.5 MB
        against roughly 0.4 MB for the crop.
        """
        apple_mask = view["apple_mask"]
        try:
            x0, y0, x1, y1 = apple_crop_box(apple_mask, self._crop_margin_px)
        except ValueError:
            return None
        cropped_image = view["image"][y0:y1, x0:x1]
        cropped_mask = apple_mask[y0:y1, x0:x1]
        return {
            "frame_index": view["frame_index"],
            "rgb_message_stamp": view["rgb_message"].header.stamp,
            "rgb_message_frame_id": str(view["rgb_message"].header.frame_id),
            "confidence": view["confidence"],
            "image": cropped_image,
            "apple_mask": cropped_mask,
            "depth_mm": view["depth_mm"][y0:y1, x0:x1],
            "ignore_mask": unmeasurable_surface_mask(cropped_image, cropped_mask),
            "camera_info": cropped_camera_info(view["camera_info"], (x0, y0, x1, y1)),
        }

    def _consider_candidate(self, stamp_ns: int, views: list[dict[str, Any]]) -> None:
        """Collect instants spread across the transit, not just the best one.

        A single instant sees about a third of the peel, so the colour ratio it
        yields swings with whichever face happens to be up. Keeping several
        instants that are far apart in time lets the roller's own tumble cover
        the rest of the surface.
        """
        session = self._session
        if session is None:
            return
        session["candidates"] = int(session["candidates"]) + 1
        position = self._group_position_m(views)
        if position is not None:
            session["last_position_m"] = position

        histogram = session["views_histogram"]
        histogram[len(views)] = histogram.get(len(views), 0) + 1
        master = self._master_position_m(views)
        if master is not None:
            session["path"].append((stamp_ns, master))

        # 면이 많을수록 한 순간이 넓게 덮지만, 요구 조건으로 걸면 세 면이 동시에
        # 잡히지 않는 동안 검사가 통째로 사라진다. 커버리지는 통과 중 회전으로
        # 채운다.
        if len(views) < self._min_views_per_instant:
            return
        instants = session["instants"]
        # 직전 순간과 너무 가까우면 자세가 거의 같아 표본이 늘지 않는다. 크롭과
        # ignore_mask 계산 전에 버려야 20Hz 부하가 사라진다.
        if instants and stamp_ns - instants[-1]["stamp_ns"] < self._min_instant_gap_ns:
            return

        sharpness = min(self._view_sharpness(view["image"]) for view in views)
        confidence = min(float(view["confidence"]) for view in views)
        cropped = [self._crop_view(view) for view in views]
        if any(item is None for item in cropped):
            return

        instants.append({
            "stamp_ns": stamp_ns,
            "score": (sharpness, confidence),
            "views": cropped,
        })
        self._thin_instants(instants)

    def _thin_instants(self, instants: list[dict[str, Any]]) -> None:
        """Drop the most redundant instant once there are too many.

        The transit length is not known while it is still happening, so instead
        of pre-computing a schedule, keep every instant until the budget is
        exceeded and then remove whichever one sits closest to its neighbours.
        That preserves the two ends and leaves the rest roughly evenly spread.
        """
        if len(instants) <= self._representative_instants:
            return
        # 양 끝은 통과 구간의 폭을 정하므로 남긴다.
        victim = min(
            range(1, len(instants) - 1),
            key=lambda i: (
                instants[i + 1]["stamp_ns"] - instants[i - 1]["stamp_ns"],
                -instants[i]["score"][0],
            ),
        )
        instants.pop(victim)

    def _finish_session(self, stamp_ns: int) -> None:
        """Publish the ROI-exit completion that starts the result deadline."""
        session = self._session
        self._session = None
        self._outside_groups = 0
        if session is None:
            return

        # 세션이 끝난 위치를 기억해 두면, 검출이 끊겼다 이어질 때 같은 사과로
        # 인식할 수 있다.
        self._last_track = {
            "apple_id": session["apple_id"],
            "stamp_ns": stamp_ns,
            "position": session.get("last_position_m"),
        }

        instants = session["instants"]
        if not instants:
            self.get_logger().warn(
                f"{session['inspection_id']}: no usable representative frame "
                f"from {session['candidates']} candidates; no result published"
            )
            return
        total_frames = sum(len(item["views"]) for item in instants)
        for item in instants:
            self._publish_group(session, item["views"], total_frames)

        completion = InspectionCompleted()
        # ROI exit is a conveyor event, so it carries the ROI frame rather
        # than any one camera optical frame.
        header = Header()
        header.stamp = self._stamp_from_ns(stamp_ns)
        header.frame_id = INSPECTION_ROI_FRAME
        completion.header = header
        completion.inspection_id = session["inspection_id"]
        completion.apple_id = session["apple_id"]
        # total_frames 는 카메라 수가 아니라 실제로 발행한 프레임 수다. 한 검사가
        # 여러 순간을 담으므로 카메라 수만으로는 수신 측이 완결을 판단할 수 없다.
        completion.total_frames = total_frames
        self._completion_publisher.publish(completion)
        span_ns = instants[-1]["stamp_ns"] - instants[0]["stamp_ns"]
        self.get_logger().info(
            f"ROI exit: {session['inspection_id']} "
            f"published={session['frames_sent']}/{total_frames} frames "
            f"({len(instants)} instants over {span_ns / 1e9:.2f}s) "
            f"{int(session.get('bytes_sent', 0)) / 1024:.0f} KB "
            f"from {session['candidates']} candidates, exit at {stamp_ns} ns"
        )
        self._log_session_diagnostics(session, instants)

    def _log_session_diagnostics(
        self, session: dict[str, Any], instants: list[dict[str, Any]]
    ) -> None:
        """Report what this transit actually yielded, in measurable terms.

        The instant budget, the minimum gap and the ROI size were all sized from
        an assumed conveyor speed. Rather than assume again, publish what the
        run itself shows: how far the apple travelled through the inspection
        region, how long that took, and how many instants came out of it.
        """
        path = session.get("path") or []
        if len(path) >= 2:
            (first_ns, first_point), (last_ns, last_point) = path[0], path[-1]
            travel_m = (
                sum((a - b) ** 2 for a, b in zip(last_point, first_point)) ** 0.5
            )
            duration_s = (last_ns - first_ns) / 1e9
            speed = f"{travel_m / duration_s:.3f} m/s" if duration_s > 0 else "n/a"
            self.get_logger().info(
                f"  transit: {travel_m * 100:.1f} cm over {duration_s:.2f}s "
                f"-> {speed} ({len(path)} positions from the master camera)"
            )

        histogram = session.get("views_histogram") or {}
        if histogram:
            spread = ", ".join(
                f"{count} view{'' if count == 1 else 's'} x{histogram[count]}"
                for count in sorted(histogram)
            )
            self.get_logger().info(f"  views in ROI per group: {spread}")

        coverage = estimated_surface_coverage(len(instants))
        if len(instants) < self._representative_instants:
            self.get_logger().warn(
                f"  only {len(instants)}/{self._representative_instants} instants; "
                f"estimated surface coverage {coverage * 100:.1f}%. The apple "
                "left the inspection region before the budget was filled: "
                "either the region is too small for this conveyor speed or the "
                "minimum instant gap is too long."
            )
        else:
            self.get_logger().info(
                f"  estimated surface coverage {coverage * 100:.1f}%"
            )

    @staticmethod
    def _stamp_from_ns(stamp_ns: int) -> Any:
        stamp = Time()
        stamp.sec = int(stamp_ns // 1_000_000_000)
        stamp.nanosec = int(stamp_ns % 1_000_000_000)
        return stamp

    def _header(self, source_stamp: Any, frame_id: str) -> Any:
        header = Header()
        header.stamp = copy.deepcopy(source_stamp)
        header.frame_id = frame_id
        return header

    @staticmethod
    def _compressed(header: Any, data: bytes, image_format: str) -> Any:
        message = CompressedImage()
        message.header = copy.deepcopy(header)
        message.format = image_format
        message.data = data
        return message

    def _publish_group(
        self,
        session: dict[str, Any],
        views: list[dict[str, Any]],
        total_frames: int,
    ) -> None:
        inspection_id = session["inspection_id"]
        apple_id = session["apple_id"]
        published_bytes = 0

        for offset, view in enumerate(views):
            # frame_index 는 검사 안에서 유일하면서 0..total_frames-1 을 빈틈없이
            #채워야 한다. InspectionFrame 이 frame_index < total_frames 를
            # 요구하기 때문이다. 예전처럼 순간 번호에 카메라 수를 곱해 매기면,
            # 어떤 순간에 면이 하나라도 빠졌을 때 total_frames 는 줄어드는데
            # 인덱스는 그대로 3씩 뛰어 뒤쪽 프레임이 전부 계약 위반으로 거부된다.
            # 어느 카메라가 찍었는지는 header.frame_id 가 이미 싣고 있으므로
            # 인덱스가 카메라를 가리킬 필요는 없다.
            frame_index = int(session["frames_sent"]) + offset
            # Each view keeps its own camera frame so the grader can tell the
            # three faces apart; the six component headers still match.
            header = self._header(
                view["rgb_message_stamp"], view["rgb_message_frame_id"]
            )
            # 크롭과 ignore_mask 는 후보를 보관할 때 이미 계산했다.
            cropped_image = view["image"]
            cropped_mask = view["apple_mask"]
            cropped_depth = view["depth_mm"]
            ignore_mask = view["ignore_mask"]

            inspection = InspectionImage()
            inspection.header = copy.deepcopy(header)
            inspection.inspection_id = inspection_id
            inspection.apple_id = apple_id
            inspection.frame_index = frame_index
            inspection.total_frames = total_frames
            inspection.image = self._compressed(
                header,
                encode_image(".jpg", cropped_image),
                "bgr8; jpeg compressed bgr8",
            )
            inspection.apple_mask = self._compressed(
                header,
                encode_image(".png", cropped_mask),
                "mono8; png",
            )
            inspection.ignore_mask = self._compressed(
                header,
                encode_image(".png", ignore_mask),
                "mono8; png",
            )
            inspection.aligned_depth = self._compressed(
                header,
                encode_image(".png", cropped_depth),
                "16UC1; compressedDepth png",
            )
            # principal point는 crop 시점에 이미 옮겨 두었다.
            inspection.camera_info = copy.deepcopy(view["camera_info"])
            inspection.camera_info.header = copy.deepcopy(header)
            published_bytes += (
                len(inspection.image.data)
                + len(inspection.apple_mask.data)
                + len(inspection.ignore_mask.data)
                + len(inspection.aligned_depth.data)
            )
            self._inspection_publisher.publish(inspection)

        # 완료 이벤트는 ROI 이탈 시점에 한 번만 낸다. 프레임 발행 때마다 내면
        # 수신 측 deadline이 매번 다시 시작된다.
        session["frames_sent"] = int(session["frames_sent"]) + len(views)
        session["bytes_sent"] = int(session.get("bytes_sent", 0)) + published_bytes
        confidences = ", ".join(f"{view['confidence']:.3f}" for view in views)
        self.get_logger().debug(
            f"{inspection_id}: apple={apple_id} views={len(views)}/{total_frames} "
            f"confidence=[{confidences}]"
        )


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source ROS 2 Jazzy and the workspace"
        ) from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node = ConveyorCameraAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
