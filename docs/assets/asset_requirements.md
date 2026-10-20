# 시뮬레이션 자산 요구사항

모든 길이는 m, 질량은 kg, 힘은 N, 토크는 N·m 단위를 사용한다. Isaac Sim
5.1.0에서 축, scale, articulation, material, collision을 검증한다.

## 로봇과 그리퍼

- Doosan M0617 6-DOF 2대
- Robotiq AGS-001-MTCP 3-finger soft gripper 2개
- 그리퍼는 rigid finger와 compliant contact를 조합해 시뮬레이션한다.
- end-effector frame: `gripper_frame`
- 물리 수확 TCP: USD `palm` 원점에서 palm local `+Y` 방향 0.0908m
- 저장 자산: `m0617_3fgripper08201638.usd`

| robot ID | 로봇 Prim | 초기 관절 자세(deg) |
|---|---|---|
| `robot_01` | `/World/Xform_01/m0617_01` | `[0, 0, -90, 0, 90, 0]` |
| `robot_02` | `/World/Xform_02/m0617_02` | `[0, 0, 90, 0, -90, 0]` |

각 로봇은 `m0617_rail/root_joint`를 Articulation root로 사용한다.

## 카메라

Intel RealSense D455를 base 카메라와 컨베이어 검사 카메라로 사용한다.

| 카메라 | USD Prim | ROS namespace | frame |
|---|---|---|---|
| robot 1 base | `/World/base_rsd455_01` | `/robot_01/base_camera` | `robot_01/base_camera` |
| robot 2 base | `/World/base_rsd455_02` | `/robot_02/base_camera` | `robot_02/base_camera` |
| conveyor top | `conv_rsd455` | `/conveyor_camera` | `quality_camera_top_optical_frame` |

RGB와 depth는 동일 render product에서 같은 해상도와 timestamp로 발행한다.
통합 기본 해상도는 1280×720, frame skip 1(약 30Hz)이다.

## 사과

- rigid body, collider, gravity 활성화
- 기준 지름 0.08m, 질량 0.3kg
- 컨베이어와 다른 사과의 물리 충돌 활성화
- 줄기에 breakable joint 적용: break force 15N, break torque 2N·m
- 작업영역을 벗어난 사과는 physics, collision, visibility를 비활성화한다.
- reset 후 재사용할 수 있도록 object pool 방식으로 관리한다.

사과 FixedJoint는 각 `apple_branch_xx` 내부에서 `branchbody`와 `applebody`를
연결한다.

## 나무와 obstacle

- 담당 영역: `/World/Xform/tree`, `/World/Xform_03/tree`
- 굵은 가지와 몸통은 PhysX collision과 planning proxy를 함께 사용한다.
- 얇은 가지와 잎은 visual/occlusion 용도로만 유지한다.
- planning proxy는 capsule, sphere, box 형상으로 단순화하고 safety margin을
  함께 기록한다.
- GPU PC 1이 reset마다 `PlanningScene` snapshot을 생성하고 Lula RRT와
  RMPflow에 같은 proxy를 적용한다.

## 컨베이어

- 2개 모듈, 전체 길이 3.3m
- 컨베이어 1: 입력·이송 모듈
- 컨베이어 2: 롤러 방식 검사 모듈
- 컨베이어 2 상부에 D455 카메라 1대를 설치해 RGB-D를 취득한다.
- 모듈 상면 높이 40~60mm, 유효 폭 0.25~0.30m
- 기본 이송 속도 0.10m/s(`conveyor_camera_publish.py`)
- 검증 운전 속도 0.10~0.40m/s
