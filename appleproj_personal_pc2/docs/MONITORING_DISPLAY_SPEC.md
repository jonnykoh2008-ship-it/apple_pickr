# 품질 모니터링 표시 사양

개인 PC 2 화면은 사과별 품질 결과와 컨베이어 이벤트를 시간 순서로 표시한다.

## 결과 카드

각 결과에 다음을 표시한다.

- `inspection_id`, `apple_id`
- `grade`, `status`
- `color_ratio`, `diameter_mm`, `confidence`
- `frames_used`, `frame_indices`
- 결과 simulation timestamp와 수신 wall timestamp

## 상태 표시

| status | 표시 |
|---|---|
| `VALID` | 정상 결과 |
| `RECHECK` | 재검사 필요 |
| `UNCLASSIFIED` | 등급 산출 불가 |
| `TIMEOUT` | 결과 deadline 초과 |
| `LATE_RESULT` | deadline 이후 도착 |
| `ID_MISMATCH` | ID 연결 오류 |
| `INSUFFICIENT_VIEWS` | 측정 프레임 부족 |

## 이벤트 목록

`CheckpointEvent`의 `ENTER`와 `EXIT`를 사과별로 묶어 현재 위치, 체류 시간,
순서 오류를 표시한다. 중복 ENTER와 ENTER 없는 EXIT는 경고로 표시한다.

## 갱신 원칙

- 동일 inspection의 최신 결과를 카드에 반영한다.
- 다른 apple ID가 연결되면 기존 카드와 분리하고 오류를 기록한다.
- `/clock`이 정지한 동안 simulation deadline과 상태 시간을 증가시키지 않는다.
- 네트워크가 끊기면 마지막 수신 시각과 연결 상태를 표시한다.
