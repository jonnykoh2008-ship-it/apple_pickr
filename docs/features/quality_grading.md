# 품질 검사

GPU PC 2가 컨베이어 카메라 영상을 받아 사과별 착색률을 계산하고
`/quality/results`로 결과를 발행한다. 개인 PC 2는 결과를 구독해 표시한다.

## 입력과 세션

- 입력: `/conveyor_camera/color/image_raw`, `/conveyor_camera/depth/image_raw`,
  `/conveyor_camera/camera_info`
- 검사 이미지: `/quality/inspection_images`
- 검사 완료: `/quality/inspection_completed`
- 결과: `/quality/results`
- 모든 timestamp: Isaac Sim `/clock`
- 모든 노드: `use_sim_time=true`
- 기본 카메라: `/conveyor_camera`의 top view
- 한 검사에는 한 사과만 존재하며 `inspection_id`와 `apple_id`를 유지한다.
- ROI는 `full_frame` 방식으로 처리하고 영상 경계에 닿은 사과는 제외한다.

## 영상 처리

1. RGB와 depth timestamp를 확인한다.
2. HSV 및 RGB 비율로 목표 적색 후보를 생성한다.
3. depth로 배경과 컨베이어 반사 영역을 제거한다.
4. morphology와 contour로 사과 mask를 만든다.
5. 유효 사과 표면에서 목표 착색 픽셀 비율을 계산한다.
6. tracker가 동일 사과의 검사 세션을 유지한다.

측정 파라미터는 `quality_grading_system/opencv_color_predictor.py`와
`conveyor_camera_adapter_node.py`에서 관리한다.

## 등급 규칙

등급은 `color_ratio` 하나로 결정한다.

| 등급 | 조건 |
|---|---|
| `HIGH` | `color_ratio >= 0.80` |
| `MEDIUM` | `0.60 <= color_ratio < 0.80` |
| `LOW` | `color_ratio < 0.60` |

경계값 0.80은 `HIGH`, 0.60은 `MEDIUM`으로 처리한다. `diameter_mm`은 함께
측정해 발행하지만 등급 산출에는 사용하지 않는다.

## QualityResult

```text
토픽: /quality/results
타입: appleproj_interfaces/msg/QualityResult
송신: GPU PC 2
수신: 개인 PC 2
QoS: Reliable, Volatile, Keep Last 10
```

정상 결과는 `status=VALID`와 유효한 `inspection_id`, `apple_id`, `grade`,
`color_ratio`, `frames_used`를 포함한다. 입력 ID가 바뀌거나 측정 프레임이
부족하면 각각 `ID_MISMATCH`, `INSUFFICIENT_VIEWS`를 발행한다.

## 시간과 결과 확정

- ROI 이탈 시각은 `/quality/inspection_completed.header.stamp`다.
- ROI 이탈 후 0.5 simulation-second 이내에 도착한 결과를 확정한다.
- 기한을 넘긴 결과는 `LATE_RESULT`로 기록한다.
- 기한까지 결과가 없으면 `TIMEOUT`을 한 번 발행한다.
- Timeline Pause에서는 deadline이 진행하지 않는다.

## 실행

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=101

ros2 run quality_grading_system conveyor_camera_adapter_node
ros2 run quality_grading_system quality_inspection_node \
  --ros-args -p model_backend:=opencv_color \
             -p grade_by:=color \
             -p min_valid_views:=1
```

## 검증

```bash
ros2 topic hz /conveyor_camera/color/image_raw
ros2 topic echo --once /quality/results
ros2 topic info -v /quality/results
```
