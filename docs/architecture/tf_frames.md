# TF 및 시뮬레이션 시간

## 권위와 transport

GPU PC 1의 Isaac Sim만 TF를 발행한다. 개인 PC 1은 영상 timestamp에 맞는 TF를
조회해 좌표를 변환하며 TF를 재발행하지 않는다.

- 동적 TF: `/tf`
- 정적 TF: `/tf_static`
- 시간: Isaac Sim `/clock`
- 모든 노드: `use_sim_time=true`

## 프레임 트리

```text
world
 ├─ robot_01/odom ─ robot_01/base_link ─ robot_01/.../link_6 ─ robot_01/palm
 │                                      └─ robot_01/base_camera
 ├─ robot_02/odom ─ robot_02/base_link ─ robot_02/.../link_6 ─ robot_02/palm
 │                                      └─ robot_02/base_camera
 ├─ quality_camera_top_optical_frame
 └─ conveyor_inspection_roi
```

실제 USD base 카메라 Prim은 `base_rsd455_01`, `base_rsd455_02`이며 ROS frame은
각 robot namespace를 붙인다. 컨베이어 top 카메라 frame은
`quality_camera_top_optical_frame`이다.

## 수확 좌표 규약

- `HarvestTarget.header.frame_id`: `world`
- `position`: world 좌표의 사과 중심
- `source_point`: 검출 카메라 좌표와 촬영 timestamp
- 개인 PC 1은 source point를 촬영 시각에 가장 가까운 TF로 world에 변환한다.
- GPU PC 1은 `link_6`를 Lula 제어 frame으로 사용하고, 물리 TCP pose는 `palm`에서
  계산한다.

## planning scene 좌표

`PlanningScene.header.frame_id`, obstacle pose, robot base pose, robot TCP pose,
RViz Marker/Path는 모두 `world` 기준이다. GPU PC 1은 동일 snapshot을 Lula RRT,
trajectory generation, RMPflow에 사용한다.

## 시간 규칙

- 센서, TF, target, checkpoint, 품질 결과 timestamp는 simulation time이다.
- Timeline Pause에서는 simulation-time timer와 deadline이 진행하지 않는다.
- 네트워크 생존 확인이 필요한 경우에만 별도의 wall-time watchdog을 사용한다.
- timestamp가 허용 오차를 벗어나거나 세대가 다르면 target을 폐기한다.
