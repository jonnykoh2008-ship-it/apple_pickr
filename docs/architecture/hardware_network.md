# 하드웨어 및 네트워크

## 장비

- RTX 5080 GPU 노트북 2대
- Ubuntu Linux 노트북 2대
- 2.5Gbps 5포트 Ethernet 스위치
- Doosan M0617 6-DOF 2대
- Robotiq AGS-001-MTCP 3-finger soft gripper 2개
- Intel RealSense D455 base 카메라와 컨베이어 상부 카메라 1대
- 2모듈 컨베이어(총 길이 3.3m, 2번 모듈 롤러 방식)

## 고정 IP

| 장비 | IP | 역할 |
|---|---|---|
| GPU PC 1 | `10.10.0.1` | Isaac Sim, 센서·물리·로봇 실행 |
| GPU PC 2 | `10.10.0.2` | 컨베이어 품질 검사 |
| 개인 PC 1 | `10.10.0.3` | 사과 검출·world target |
| 개인 PC 2 | `10.10.0.4` | 모니터링 |

유선 ROS 2 네트워크는 인터넷과 분리하고, 인터넷은 공용 Wi-Fi를 사용한다.

## ROS 2 운용값

- ROS 2 Jazzy
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `ROS_DOMAIN_ID=101`
- 모든 노드 `use_sim_time=true`
- Isaac Sim이 `/clock`을 단일 발행
- Fast DDS UDP transport 사용

모든 PC의 domain, RMW, interface 빌드 revision을 동일하게 유지한다. Wi-Fi와
Ethernet의 라우팅 우선순위를 고정해 DDS discovery가 유선망을 사용하도록 한다.

## 데이터 경로

GPU PC 1은 base/conveyor RGB-D, CameraInfo, TF, `/clock`, planning scene과
checkpoint를 발행한다. 개인 PC 1은 base RGB-D에서 target을 계산해 GPU PC 1으로
반환한다. GPU PC 2는 컨베이어 raw 영상을 검사 메시지로 묶어 품질 결과를 발행하고,
개인 PC 2는 결과와 checkpoint를 구독한다.

## QoS

- RGB, depth, CameraInfo: Reliable, Volatile, Keep Last 6
- target과 motion status: Reliable, Volatile, Keep Last 10
- quality result와 checkpoint: Reliable, Volatile, Keep Last 10
- `/simulation/state`, `/planning_scene`: Reliable, Transient Local, Keep Last 1

영상 전송 지연, target timestamp 오차, 품질 결과 latency와 메시지 손실률은
`ros2 topic hz`와 `ros2 topic info -v`로 통합 시험 때 확인한다.
