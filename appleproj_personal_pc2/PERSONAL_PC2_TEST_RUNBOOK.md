# 개인 PC 2 테스트 런북

개인 PC 2의 `quality_monitor`가 품질 결과와 checkpoint를 정상적으로 수신하고,
ID·상태·deadline을 올바르게 처리하는지 검증한다.

## 사전 조건

```bash
cd /home/rokey/cobot3_ws/rokey_isaacsim_project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=101
```

## 실행

```bash
ros2 launch appleproj_personal_pc2 personal_pc2.launch.py
```

실제 카메라 ROI 종료를 deadline 기준으로 사용할 때는 checkpoint ID를 전달한다.

```bash
ros2 launch appleproj_personal_pc2 personal_pc2.launch.py \
  deadline_checkpoint_id:=<ROI_EXIT_CHECKPOINT_ID>
```

## 토픽 확인

```bash
ros2 topic info -v /quality/results
ros2 topic info -v /conveyor/checkpoint_events
ros2 topic echo --once /quality/results
ros2 topic echo --once /conveyor/checkpoint_events
```

## 오프라인 메시지 시험

개발용 mock source로 모니터 노드의 상태 처리를 확인할 수 있다.

```bash
python3 -m appleproj_personal_pc2.mock_quality_source \
  --inspection-id inspection_001 \
  --apple-id apple_001 \
  --grade HIGH \
  --status VALID
```

mock source와 실제 Isaac Sim publisher는 동시에 실행하지 않는다.

## 검증 시나리오

1. 정상 `VALID` 결과가 ID와 측정값을 포함해 표시되는지 확인한다.
2. 같은 inspection에 다른 apple ID가 들어오면 `ID_MISMATCH`가 되는지 확인한다.
3. `ENTER → EXIT` 순서가 정상 기록되는지 확인한다.
4. 중복 ENTER와 ENTER 없는 EXIT가 경고로 표시되는지 확인한다.
5. ROI 이탈 후 0.5 simulation-second 안의 결과가 정상 확정되는지 확인한다.
6. deadline 이후 결과가 `LATE_RESULT`로 기록되는지 확인한다.
7. 결과가 없을 때 `TIMEOUT`이 한 번만 기록되는지 확인한다.
8. Timeline Pause 중 deadline이 진행하지 않는지 확인한다.

## 단위 테스트

```bash
colcon test --packages-select appleproj_personal_pc2
colcon test-result --verbose
```

## 합격 기준

- 정상·오류 결과가 구분되어 표시된다.
- 사과별 inspection ID와 checkpoint ID 연결이 보존된다.
- 중복·순서 오류가 다른 사과의 상태를 변경하지 않는다.
- simulation time과 수신 wall time이 별도로 기록된다.
- 모니터 노드가 메시지 오류로 종료되지 않는다.
