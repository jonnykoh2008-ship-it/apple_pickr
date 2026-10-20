# 사과 수확 및 로봇 동작

GPU PC 1이 개인 PC 1의 `HarvestTarget`을 검증하고, Lula 기반 경로 계획과
M0617 실행을 한 번의 Action lifecycle로 관리한다.

## 동작 단계

```text
APPROACH → GRASP → TWIST → PULL → TRANSPORT → PLACE → RELEASE → RETRACT
```

- `APPROACH`: transit, staging, pre-grasp 경로를 Lula RRT로 계획한다.
- `GRASP`: palm 접촉을 확인한 뒤 세 손가락을 폐합한다.
- `TWIST`: 손목을 45° 회전한다(1초).
- `PULL`: 줄기 반대 방향으로 50mm/s, 최대 100mm를 직선 당김한다.
- `TRANSPORT`: 컨베이어 1 투입 pose로 이동한다.
- `PLACE`/`RELEASE`: 벨트 상면 30mm 이내에서 개방한다.
- `RETRACT`: 안전 경로로 후퇴해 다음 target을 기다린다.

## 계획과 실행

- LulaKinematicsSolver로 IK를 계산한다.
- Lula RRT는 transit/staging/pre-grasp/retract의 전역 c-space 경로에 사용한다.
- RRT waypoint는 시간 매개화 trajectory로 변환한 뒤 RMPflow로 추종한다.
- planning scene은 GPU PC 1이 생성하며 몸통·굵은 가지 proxy와 safety margin을
  RRT와 RMPflow에 동일하게 적용한다.
- GPU PC 1은 실행 중 PhysX contact를 감시하고 예기치 않은 로봇-나무 접촉 시
  Action을 정지한다.

## 파지·줄기 분리

- 물리 수확 TCP는 USD `palm` 기준으로 계산한다.
- 손바닥 collider 접촉을 먼저 확인하고 손가락 collider를 폐합한다.
- 줄기 FixedJoint의 break force는 15N, break torque는 2N·m다.
- `PULL` 성공은 stem joint 분리와 TCP 진전을 함께 확인한다.

## Action 계약

```text
액션: /<robot_id>/harvest/robot_motion
타입: appleproj_interfaces/action/RobotMotion
```

Goal에는 `motion_type`, `target_pose`, `reset_id`, `scene_version`, `waypoints`를
넣는다. 실행 중에는 동일 로봇의 새 Goal을 받지 않고 cancel만 허용한다.
Feedback의 `progress`는 0.0~1.0이며 성공 Result의 `error_code`는 빈 문자열이다.

## 실패 처리

- IK 또는 경로 실패: `APPROACH_UNREACHABLE`
- 예기치 않은 접촉: `UNEXPECTED_CONTACT`
- 줄기 미분리: `PULL_TIMEOUT`
- target 세대·timestamp 불일치: `STALE_TARGET`
- Action cancel: 즉시 정지 후 취소 결과 반환
- 실패 시 로봇을 정지하고 `MotionStatus`에 원인과 메시지를 발행한다.

## 다중 사과

- target은 `target_id`별로 저장하고 robot base에서 가까운 순서로 실행한다.
- 실행 중인 target의 위치 갱신은 현재 Goal을 바꾸지 않는다.
- 아직 시작하지 않은 target은 같은 ID의 최신 timestamp로 갱신한다.
- 성공 또는 최종 실패 target은 같은 `reset_id`에서 재실행하지 않는다.

## Reset

Timeline Stop/Reset 시 active Goal, target queue, RRT tree, trajectory와 완료 기록을
폐기한다. `SimulationState.reset_id`를 증가시킨 뒤 새 planning scene snapshot을
발행하고, 개인 PC 1은 새 `READY`/`PLAYING` 상태를 확인한 후 target을 재개한다.
