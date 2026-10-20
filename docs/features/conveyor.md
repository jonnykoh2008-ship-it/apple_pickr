# 컨베이어

컨베이어 물리, checkpoint와 raw 카메라 스트림은 GPU PC 1의 Isaac Sim이 소유한다.
GPU PC 2는 raw 스트림을 구독해 사과를 추적하고 검사 프레임을 만든다. 개인 PC 2는
checkpoint와 품질 결과를 모니터링한다.

## 공통 사양

- 모듈 수: 2개
- 전체 길이: 3.3m
- 1번 모듈: 입력·이송
- 2번 모듈: 롤러 방식 검사 구간
- 유효 폭: 0.25~0.30m
- 측면 가이드 안쪽 폭: 0.14~0.16m
- 롤러/벨트 상면 높이: 40~60mm
- 기본 속도: 0.10m/s
- 검증 속도 범위: 0.10~0.40m/s
- 모든 사과는 rigid body와 collider를 갖는다.

## 컨베이어 1: 입력·이송

- 입력·이송용 모듈
- 로봇은 시작점에서 0.15~0.20m 지점에 사과를 투입한다.
- 벨트 상면에서 30mm 이하 높이로 배치한다.
- 중심선 기준 좌우 배치 오차 목표는 ±30mm다.

## 컨베이어 2: 롤러 검사

- 롤러 방식 모듈
- 전체 3.3m 중 2번 모듈을 검사 구간으로 사용한다.
- 롤러의 surface velocity와 마찰로 사과를 이송·회전한다.
- 모듈 상부에 D455 카메라 1대를 설치한다.
- 상부 카메라의 RGB-D 스트림만 품질 검사에 사용한다.
- GPU PC 1은 `/conveyor_camera/{color,image_raw,depth,image_raw,camera_info}`와
  `/tf_static`을 발행한다.
- GPU PC 2의 adapter는 `roi_mode=full_frame`과 3D tracker로 검사 세션을
  관리하고 `/quality/inspection_images`를 발행한다.
- 검사 세션은 동일 `apple_id`를 유지하고 ROI 이탈 시
  `/quality/inspection_completed`를 발행한다.
- 화면 경계에 닿은 사과는 측정에서 제외한다.

## 검사 완료 및 라인 끝 배출

- `apple_id`와 `/quality/results`의 연결 상태를 확인한다.
- 정상 결과와 예외 결과 모두 라인 끝으로 이동시킨다.
- `CheckpointEvent`의 `ENTER`·`EXIT`로 공정 순서와 점유 시간을 기록한다.

## 검사 프레임과 품질 계산

- RGB, depth, CameraInfo timestamp는 동일해야 한다.
- adapter 기본 카메라 namespace는 `/conveyor_camera`다.
- adapter는 추적 위치의 연속성으로 같은 사과의 세션을 재사용한다.
- 결과 deadline은 ROI 이탈 후 0.5 simulation-second다.
- deadline 전 결과는 `VALID` 또는 품질 상태로 확정하고, 이후 결과는
  `LATE_RESULT`로 기록한다.
- 착색률은 모든 유효 픽셀의 목표 착색 픽셀 수를 유효 사과 표면 픽셀 수로
  나눈 값이다.

## 시간 및 정지

- 이송 timer와 품질 deadline은 `/clock`을 사용한다.
- Timeline Pause에서는 timer와 deadline이 정지한다.
- Stop/Reset 시 GPU PC 1은 컨베이어 실행 context와 사과 대기열을 초기화하고
  새 `SimulationState.reset_id`를 발행한다.
- 통신 장애가 감지되면 stale target과 stale 품질 결과를 재사용하지 않는다.
