# 모니터링 이벤트 이력

개인 PC 2는 품질 결과와 컨베이어 이벤트를 JSON Lines로 기록해 사후 분석에
사용한다.

## 레코드 형식

한 줄에 하나의 UTF-8 JSON object를 저장한다.

```json
{
  "event_type": "QUALITY_RESULT",
  "simulation_time_ns": 1234567890,
  "received_wall_time_ns": 1234567890123456,
  "payload": {}
}
```

`simulation_time_ns`는 공정 순서 분석, `received_wall_time_ns`는 네트워크 지연과
노드 생존 분석에 사용한다.

## event_type

- `QUALITY_RESULT`
- `CHECKPOINT_EVENT`
- `MONITOR_NOTICE`
- `RETRY_REQUEST`
- `RETRY_RESPONSE`

## 기록 규칙

- 저장 경로는 실행자가 지정한다.
- 기존 파일은 append만 하며 truncate하지 않는다.
- 빈 event type, 음수 simulation time, JSON 직렬화 불가 payload는 거부한다.
- `JsonlEventHistory`가 한 레코드씩 append한다.

## 검증

```bash
pytest -q appleproj_personal_pc2/test/test_event_history.py
```
