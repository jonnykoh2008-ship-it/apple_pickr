# 수확용 인식

개인 PC 1이 GPU PC 1의 base RGB-D와 CameraInfo를 받아 사과 중심을 계산하고,
`world` 좌표의 `HarvestTarget`을 발행한다. GPU PC 1은 target을 검증한 뒤 로봇
계획과 실행을 담당한다.

## 카메라·토픽

| 로봇 | USD 카메라 | 입력 토픽 |
|---|---|---|
| `robot_01` | `/World/base_rsd455_01` | `/robot_01/base_camera/{color,depth,image_raw,camera_info}` |
| `robot_02` | `/World/base_rsd455_02` | `/robot_02/base_camera/{color,depth,image_raw,camera_info}` |

관련 TF frame은 `robot_01/base_camera`, `robot_02/base_camera`다.

## 인식 흐름

```text
RGB + depth + CameraInfo
  → RGB/depth timestamp 동기화
  → HSV 전경·contour 검출
  → depth 유효성 검사
  → camera 좌표 중심 계산
  → timestamp에 맞는 TF로 world 변환
  → HarvestTarget 발행
```

기준값은 다음과 같다.

- 유효 depth 범위: 0.30~6.00m
- 검출 영역 유효 depth 비율: 0.30 이상
- RGB/depth 동기화 오차: 0.05초 이하
- TF timestamp 오차: 0.20초 이하
- 검출 confidence: 0.30 이상
- track 연결 반경: 0.10m

## HarvestTarget

```text
토픽: /<robot_id>/harvest/target
타입: appleproj_interfaces/msg/HarvestTarget
송신: 개인 PC 1
수신: GPU PC 1
```

`header.frame_id`는 `world`다. `target_id`, `reset_id`, `scene_version`,
`position`, `source_point`, `confidence`, `valid_depth_ratio`,
`tf_time_error_sec`를 함께 발행한다. GPU PC 1은 현재 `SimulationState`가
`READY` 또는 `PLAYING`이고 target의 세대·timestamp가 유효할 때만 계획한다.

## 다중 로봇과 ID

- `--robot-id robot_01`과 `--robot-id robot_02`가 토픽 namespace와 TF frame을
  결정한다.
- target queue와 실행 상태는 robot별로 분리한다.
- 같은 `reset_id`에서 동일 target ID의 최신 timestamp만 유지한다.
- reset 시 detector의 track cache와 target queue를 폐기한다.

## 장애물 책임

나무·가지 planning proxy 생성과 안전거리 적용은 GPU PC 1이 담당한다. 개인 PC 1은
사과 target과 인식 상태만 전달하며 planner 입력을 승인하지 않는다.

## 인식 상태

유효 target을 만들 수 없을 때 `HarvestPerceptionStatus.status`로
`NO_DETECTION`, `DEPTH_INVALID`, `TF_UNAVAILABLE`, `STALE_FRAME`,
`LOW_CONFIDENCE`, `RESET_MISMATCH`, `SIMULATION_NOT_READY`,
`INPUT_NOT_SYNCHRONIZED`, `INTERNAL_ERROR`를 발행한다.
