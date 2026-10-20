# 시스템 개요

## 목적

하나의 Isaac Sim World에서 두 개의 M0617 로봇이 담당 수확 영역의 사과를 수확해
공용 컨베이어로 투입하고, GPU PC 2가 착색률을 계산해 품질 등급을 발행한다.

## 전체 흐름

```text
GPU PC 1 Isaac Sim
  RGB-D/CameraInfo/TF + /clock
       │
       ├──► 개인 PC 1: 사과 검출·depth projection
       │       └── /<robot_id>/harvest/target ──► GPU PC 1
       │
       └──► GPU PC 2: 컨베이어 ROI/tracker·검사 프레임 구성
               └── /quality/results ──► 개인 PC 2 모니터
```

수확과 품질 검사는 기능적으로 분리한다. 수확 계획·물리 실행·안전 판정은 GPU
PC 1이 단독으로 결정하고, 품질 등급 계산은 GPU PC 2가 담당한다.

컨베이어는 2개 모듈, 총 길이 3.3m로 구성한다. 1번 모듈은 입력·이송용이고,
2번 모듈은 롤러 방식의 검사 구간이다. 2번 모듈 상부의 D455 카메라 1대가 RGB-D를
취득한다.

## PC별 책임

- **GPU PC 1**: Isaac Sim, PhysX, 센서, TF, `/clock`, planning scene, Lula RRT,
  trajectory generation, RMPflow, 로봇 Action, 컨베이어 이송과 checkpoint.
- **GPU PC 2**: 컨베이어 RGB-D 수신, ROI/tracker, 검사 프레임 구성, 착색률 계산,
  `QualityResult` 통합 발행.
- **개인 PC 1**: RGB-D 사과 검출, depth projection, `world` 좌표 target 발행,
  RViz 원격 표시.
- **개인 PC 2**: 품질 결과와 checkpoint 수신, ID·상태·deadline 모니터링, 운영 로그.

## 로봇·USD 구성

| robot ID | 로봇 Prim | 초기 관절 자세(deg) | 담당 영역 | base 카메라 |
|---|---|---|---|---|
| `robot_01` | `/World/Xform_01/m0617_01` | `[0, 0, -90, 0, 90, 0]` | `/World/Xform` | `/World/base_rsd455_01` |
| `robot_02` | `/World/Xform_02/m0617_02` | `[0, 0, 90, 0, -90, 0]` | `/World/Xform_03` | `/World/base_rsd455_02` |

각 로봇의 Articulation root는 `m0617_rail/root_joint`다. 사과 FixedJoint는
각 `apple_branch_xx` 내부에서 `branchbody`와 `applebody`를 연결한다.

## 품질 등급

등급은 유효 표면의 착색 픽셀 비율인 `color_ratio`로 결정한다.

| 등급 | 조건 |
|---|---|
| `HIGH` | `color_ratio >= 0.80` |
| `MEDIUM` | `0.60 <= color_ratio < 0.80` |
| `LOW` | `color_ratio < 0.60` |

직경은 `diameter_mm` 측정값으로만 전달한다.

## 시간·안전

- 모든 header와 deadline은 Isaac Sim `/clock`을 기준으로 한다.
- 모든 ROS 2 노드는 `use_sim_time=true`를 사용한다.
- Timeline이 Stop/Reset되면 GPU PC 1은 실행 context와 target queue를 폐기하고
  `reset_id`를 갱신한다.
- stale `reset_id`·`scene_version` target은 실행하지 않는다.
- RViz는 시각화 전용이며 모션 실행의 권위자가 아니다.
