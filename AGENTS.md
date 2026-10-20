# Repository Working Rules

## Approval gate

- 파일이나 소스 코드를 생성·수정·이동·삭제하기 전에 사용자의 명시적 승인을 받는다.
- 분석·검토·계획·읽기 요청과 테스트 실행은 파일 수정 승인으로 간주하지 않는다.
- 승인된 파일과 범위를 넘어서는 변경은 추가 승인을 받는다.
- 수정 후 변경 파일과 수행한 검증을 보고한다.

## Project baseline

- Target simulator: NVIDIA Isaac Sim 5.1.0
- Target middleware: ROS 2 Jazzy on Ubuntu 24.04 using Fast DDS
- 모든 ROS 2 노드는 Isaac Sim `/clock`과 `use_sim_time=true`를 사용한다.
- 기능 문서는 `docs/features`, 아키텍처 문서는 `docs/architecture`에서 관리한다.
- 실행에 필요한 값과 인터페이스는 관련 문서와 코드에서 같은 값으로 유지한다.

## PC ownership and edit boundaries

| Owner | Responsibilities | Current owned source examples |
|---|---|---|
| GPU PC 1 | Isaac Sim, physics, RGB-D/CameraInfo/TF, planning scene, Lula RRT/trajectory/RMPflow, robot Action, conveyor runtime and safety | `apple_pick.py`, `vision_apple_pick.py`, `base_camera_publish.py`, `conveyor_camera_publish.py`, `harvest_coordinator.py` |
| GPU PC 2 | Conveyor-camera subscription, ROI/tracker, inspection frame collection, quality inference and result integration | `quality_grading_system/` |
| Personal PC 1 | RGB-D subscription, apple detection, depth projection, world target publication and RViz | `base_apple_detector.py`, `harvest_multi_robot.launch.py` |
| Personal PC 2 | Quality-result and checkpoint monitoring, display and event history | `appleproj_personal_pc2/` |

공유 인터페이스와 문서는 `appleproj_interfaces/`, `docs/`, `README.md`,
`requirements.txt`, `AGENTS.md`, `CLAUDE.md`다. 공유 파일을 수정할 때는 영향을
받는 PC를 보고한다.

소스 수정 전 실행 PC와 대상 PC의 소유권을 확인한다. 다른 PC 소유 소스의 변경이
필요하면 직접 수정하지 않고 대상 파일·함수, 문제, 제안 동작, 인터페이스 영향,
검증 절차를 포함한 변경 검토를 요청한다.

## Required documentation routing

| Task scope | Required documents |
|---|---|
| Project-wide architecture | `README.md`, `docs/architecture/system_overview.md` |
| Apple or obstacle perception | `docs/features/harvest_perception.md`, `docs/architecture/tf_frames.md`, `docs/architecture/ros2_interfaces.md` |
| Robot motion or grasping | `docs/features/harvesting.md`, `docs/assets/asset_requirements.md`, `docs/architecture/tf_frames.md` |
| Conveyor implementation | `docs/features/conveyor.md`, `docs/assets/asset_requirements.md`, `docs/architecture/ros2_interfaces.md` |
| Quality inspection or grading | `docs/features/quality_grading.md`, `docs/features/conveyor.md`, `docs/architecture/ros2_interfaces.md` |
| Rail or multi-tree integration | `docs/architecture/system_overview.md`, `docs/architecture/tf_frames.md`, `docs/assets/asset_requirements.md` |
| ROS 2 topics, messages or QoS | `docs/architecture/ros2_interfaces.md`, `docs/architecture/tf_frames.md` |
| Multi-PC or network setup | `docs/architecture/hardware_network.md`, `docs/architecture/ros2_interfaces.md` |
| Robot, camera, apple, tree or conveyor assets | `docs/assets/asset_requirements.md` and the related feature document |
| TF, timestamp or simulation time | `docs/architecture/tf_frames.md`, `docs/architecture/ros2_interfaces.md` |

아키텍처 문서는 인터페이스·TF·시간·네트워크의 기준이며, 기능 문서는 동작 규칙의
기준이다. 문서와 코드가 다르면 변경 전에 차이를 보고하고 승인된 값으로 함께
갱신한다.
