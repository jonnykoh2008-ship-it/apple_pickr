# Repository Working Rules

이 저장소는 [AGENTS.md](AGENTS.md)의 작업 규칙을 따른다.

## 기본 환경

- NVIDIA Isaac Sim 5.1.0
- Ubuntu 24.04 / ROS 2 Jazzy / Fast DDS
- Isaac Sim `/clock`과 `use_sim_time=true`
- 통합 ROS domain `101`

## PC 소유권

- GPU PC 1: Isaac Sim, 물리, 센서·TF, planning scene, 로봇·컨베이어 실행
- GPU PC 2: 컨베이어 영상과 품질 추론
- 개인 PC 1: RGB-D 사과 인식과 world target
- 개인 PC 2: 품질 결과와 checkpoint 모니터링

공유 파일은 `appleproj_interfaces/`, `docs/`, `README.md`, `requirements.txt`,
`AGENTS.md`, `CLAUDE.md`다. 파일을 수정하기 전에 사용자의 승인을 받고, 수정 후
영향받는 PC와 검증 결과를 보고한다.

## 문서 기준

- 시스템 구조: `docs/architecture/system_overview.md`
- 통신·QoS: `docs/architecture/ros2_interfaces.md`
- 네트워크: `docs/architecture/hardware_network.md`
- TF·시간: `docs/architecture/tf_frames.md`
- 수확: `docs/features/harvest_perception.md`, `docs/features/harvesting.md`
- 컨베이어: `docs/features/conveyor.md`
- 품질 검사: `docs/features/quality_grading.md`

아키텍처 문서는 인터페이스와 시스템 경계의 기준이며, 기능 문서는 각 노드의
동작 기준이다. 문서와 코드가 다르면 변경 전에 차이를 확인한다.
