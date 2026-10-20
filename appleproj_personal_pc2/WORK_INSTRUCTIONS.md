# 개인 PC 2 운영 지침

개인 PC 2는 GPU PC 2의 품질 결과와 GPU PC 1의 컨베이어 checkpoint를 받아
사과별 상태를 표시하고 운영 로그를 기록한다.

## 기준 환경

- Ubuntu 24.04
- ROS 2 Jazzy / Fast DDS
- `ROS_DOMAIN_ID=101`
- 모든 노드 `use_sim_time=true`
- 공정 시간은 Isaac Sim `/clock`, 통신 생존 확인은 wall time

## 입력

| 이름 | 타입 | 용도 |
|---|---|---|
| `/quality/results` | `appleproj_interfaces/msg/QualityResult` | 등급·측정값 표시 |
| `/conveyor/checkpoint_events` | `appleproj_interfaces/msg/CheckpointEvent` | ENTER/EXIT 순서와 ID 확인 |
| `/clock` | `rosgraph_msgs/msg/Clock` | deadline 기준 |

## 처리 규칙

- `inspection_id`와 `apple_id`가 비어 있으면 메시지를 거부한다.
- 동일 `inspection_id`에 다른 `apple_id`가 들어오면 `ID_MISMATCH`로 표시한다.
- checkpoint event는 `ENTER`와 `EXIT`만 허용한다.
- 동일 checkpoint의 중복 ENTER와 ENTER 없는 EXIT를 경고한다.
- ROI 이탈 후 0.5 simulation-second까지 결과를 기다린다.
- 기한 전 결과는 정상 처리하고, 기한 이후 결과는 `LATE_RESULT`로 표시한다.
- 기한까지 결과가 없으면 `TIMEOUT`을 한 번 기록한다.
- Timeline Pause 중에는 simulation deadline을 진행하지 않는다.

## 결과 표시

다음 필드를 화면과 로그에 표시한다.

- `inspection_id`, `apple_id`
- `grade`, `status`, `confidence`, `color_ratio`
- `diameter_mm`, `frames_used`, `frame_indices`
- 결과 timestamp와 수신 wall timestamp

등급은 `HIGH`(착색률 80% 이상), `MEDIUM`(60% 이상 80% 미만), `LOW`(60% 미만)로
표시한다. 직경은 측정값으로만 표시한다.

## 실행

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=101 ros2 launch appleproj_personal_pc2 personal_pc2.launch.py
```

## 운영 로그

`event_history.py`의 JSON Lines 기록기는 다음 필드를 저장한다.

- `event_type`
- `simulation_time_ns`
- `received_wall_time_ns`
- `payload`

운영 로그 경로는 실행자가 명시한 경로를 사용하며 UTF-8로 기록한다.

## 장애 처리

- 품질 결과와 checkpoint의 ID가 맞지 않으면 해당 사과를 오류 상태로 표시한다.
- `/clock`이 멈추면 deadline과 상태 전이를 일시 정지한다.
- ROS 연결이 끊기면 마지막 수신 시각을 경고하고 새 결과를 확정하지 않는다.
- 노드 재시작 후에는 새 메시지부터 상태를 구성한다.
