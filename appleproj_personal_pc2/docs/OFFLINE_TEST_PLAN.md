# 개인 PC 2 오프라인 시험 계획

GPU PC 1과 GPU PC 2의 실제 publisher 없이 개인 PC 2의 수신·표시·상태 관리를
검증한다.

## 원칙

- mock publisher는 기능 확인용으로만 사용한다.
- ROS 2 domain은 `101`을 사용한다.
- `/clock`을 제공하지 않는 환경에서는 simulation deadline을 검증하지 않는다.
- 실제 publisher와 mock publisher를 동시에 실행하지 않는다.

## 시험 순서

1. 개인 PC 2 모니터를 실행한다.
2. `mock_quality_source`로 정상 `QualityResult`를 한 건 발행한다.
3. 모니터 로그에서 ID, 등급, 상태, 착색률을 확인한다.
4. `CheckpointEvent`의 ENTER와 EXIT를 발행해 순서를 확인한다.
5. 다른 apple ID, 중복 ENTER, ENTER 없는 EXIT를 발행해 경고를 확인한다.
6. 시험 후 mock publisher를 종료한다.

## 합격 기준

- 정상 결과가 `VALID`로 표시된다.
- ID 불일치가 `ID_MISMATCH`로 표시된다.
- checkpoint 순서 오류가 경고로 표시된다.
- 중복 메시지가 별도로 기록된다.
- 모니터 노드가 입력 오류로 종료되지 않는다.
