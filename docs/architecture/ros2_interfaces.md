# ROS 2 인터페이스

이 문서는 네 PC가 공유하는 현재 ROS 2 데이터 계약이다.

## 공통 규칙

- 패키지: `appleproj_interfaces`
- 시간: Isaac Sim `/clock`, 모든 노드 `use_sim_time=true`
- DDS: Fast DDS, `ROS_DOMAIN_ID=101`
- 로봇 namespace: `/robot_01`, `/robot_02`
- `/clock`, `/simulation/state`, `/planning_scene`, `/quality/*`, `/conveyor/*`는
  global topic이다.
- TF transport는 `/tf`, `/tf_static`이며 frame ID에는 robot prefix를 사용한다.

## 인터페이스 목록

| 종류 | 이름 | 타입 | 송신 | 수신 |
|---|---|---|---|---|
| Topic | `/<robot_id>/base_camera/color/image_raw` | `sensor_msgs/msg/Image` | GPU PC 1 | 개인 PC 1 |
| Topic | `/<robot_id>/base_camera/depth/image_raw` | `sensor_msgs/msg/Image` | GPU PC 1 | 개인 PC 1 |
| Topic | `/<robot_id>/base_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | GPU PC 1 | 개인 PC 1 |
| Topic | `/<robot_id>/harvest/target` | `appleproj_interfaces/msg/HarvestTarget` | 개인 PC 1 | GPU PC 1 |
| Topic | `/<robot_id>/harvest/perception_status` | `appleproj_interfaces/msg/HarvestPerceptionStatus` | 개인 PC 1 | GPU PC 1 |
| Action | `/<robot_id>/harvest/robot_motion` | `appleproj_interfaces/action/RobotMotion` | GPU PC 1 내부 | GPU PC 1 내부 |
| Topic | `/<robot_id>/harvest/motion_status` | `appleproj_interfaces/msg/MotionStatus` | GPU PC 1 | 모니터 |
| Topic | `/simulation/state` | `appleproj_interfaces/msg/SimulationState` | GPU PC 1 | 전체 |
| Topic | `/planning_scene` | `appleproj_interfaces/msg/PlanningScene` | GPU PC 1 | 개인 PC 1 |
| Service | `/planning_scene/get_snapshot` | `appleproj_interfaces/srv/GetPlanningScene` | GPU PC 1 | 디버그 |
| Service | `/conveyor/place_command` | `appleproj_interfaces/srv/PlaceCommand` | GPU PC 1 | GPU PC 1 내부 |
| Topic | `/conveyor_camera/color/image_raw` | `sensor_msgs/msg/Image` | GPU PC 1 | GPU PC 2 |
| Topic | `/conveyor_camera/depth/image_raw` | `sensor_msgs/msg/Image` | GPU PC 1 | GPU PC 2 |
| Topic | `/conveyor_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | GPU PC 1 | GPU PC 2 |
| Topic | `/quality/inspection_images` | `appleproj_interfaces/msg/InspectionImage` | GPU PC 2 adapter | GPU PC 2 검사 노드 |
| Topic | `/quality/inspection_completed` | `appleproj_interfaces/msg/InspectionCompleted` | GPU PC 2 adapter | GPU PC 2 검사 노드 |
| Topic | `/quality/results` | `appleproj_interfaces/msg/QualityResult` | GPU PC 2 | 개인 PC 2 |
| Topic | `/conveyor/checkpoint_events` | `appleproj_interfaces/msg/CheckpointEvent` | GPU PC 1 | 개인 PC 2 |
| Topic | `/clock` | `rosgraph_msgs/msg/Clock` | Isaac Sim | 전체 |
| Topic | `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | GPU PC 1 | 개인 PC 1/RViz |

## QoS

| 데이터 | Reliability | Durability | History |
|---|---|---|---|
| RGB/depth/CameraInfo | Reliable | Volatile | Keep Last 6 |
| HarvestTarget/MotionStatus | Reliable | Volatile | Keep Last 10 |
| QualityResult/CheckpointEvent | Reliable | Volatile | Keep Last 10 |
| SimulationState/PlanningScene | Reliable | Transient Local | Keep Last 1 |

## HarvestTarget

`HarvestTarget.header.frame_id`는 `world`이며, position은 world 좌표의 사과 중심이다.
`source_point`는 검출 카메라 좌표를 보존한다. `reset_id`와 `scene_version`은
target 생성 당시의 시뮬레이션 세대다. GPU PC 1은 현재 세대와 timestamp, confidence,
valid depth ratio, TF 오차를 검증한 뒤 계획한다.

## QualityResult

`grade`는 `color_ratio`로 계산한다.

- `HIGH`: `color_ratio >= 0.80`
- `MEDIUM`: `0.60 <= color_ratio < 0.80`
- `LOW`: `color_ratio < 0.60`

`diameter_mm`은 측정값이고 등급 계산에는 사용하지 않는다. `status`는
`VALID`, `RECHECK`, `UNCLASSIFIED`, `TIMEOUT`, `LATE_RESULT`, `ID_MISMATCH`,
`INSUFFICIENT_VIEWS` 중 하나다. 정상 결과의 `error_code`는 빈 문자열을 사용한다.

## CheckpointEvent

`event`는 `ENTER=1` 또는 `EXIT=2`다. `apple_id`와 `checkpoint_id`는 비어 있지
않아야 하며 개인 PC 2는 사과별 ENTER/EXIT 순서를 기록한다.

## SimulationState

`state`는 `STOPPED`, `INITIALIZING`, `READY`, `PLAYING`, `PAUSED` 중 하나다.
GPU PC 1은 `READY` 또는 `PLAYING` 상태에서만 target을 실행한다. Stop/Reset 시
`reset_id`를 증가시키고 이전 실행 context를 폐기한다.

## PlanningScene

`header.frame_id`, `robot_base_pose`, `robot_tcp_pose`, obstacle pose는 모두
`world`다. `ObstacleProxy`는 몸통·굵은 가지의 sphere/box/capsule 형상과
`safety_margin`을 전달한다. RRT와 RMPflow는 동일 snapshot을 사용한다.
