"""GPU PC 1 수확 파이프라인 일괄 실행 런치.

robot_01/robot_02 각각의 base_apple_detector와 harvest_coordinator(--execute)
네 프로세스를 ROS_DOMAIN_ID=103으로 한 번에 띄운다.

사용법:
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    ros2 launch harvest_multi_robot.launch.py

옵션:
    ros2 launch harvest_multi_robot.launch.py domain_id:=103 execute:=true
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ROBOT_IDS = ("robot_01", "robot_02")


def generate_launch_description() -> LaunchDescription:
    domain_id = LaunchConfiguration("domain_id")
    execute = LaunchConfiguration("execute")

    actions = [
        DeclareLaunchArgument("domain_id", default_value="103"),
        DeclareLaunchArgument(
            "execute",
            default_value="true",
            description="harvest_coordinator에 --execute를 붙일지 여부",
        ),
    ]

    env = {"ROS_DOMAIN_ID": domain_id}

    for robot_id in ROBOT_IDS:
        actions.append(
            ExecuteProcess(
                cmd=[
                    "python3",
                    os.path.join(PROJECT_DIR, "base_apple_detector.py"),
                    "--robot-id",
                    robot_id,
                ],
                name=f"base_apple_detector_{robot_id}",
                additional_env=env,
                output="screen",
            )
        )
        actions.append(
            ExecuteProcess(
                cmd=[
                    "python3",
                    os.path.join(PROJECT_DIR, "harvest_coordinator.py"),
                    "--robot-id",
                    robot_id,
                    "--execute",
                ],
                name=f"harvest_coordinator_{robot_id}",
                additional_env=env,
                output="screen",
                condition=IfCondition(execute),
            )
        )
        actions.append(
            ExecuteProcess(
                cmd=[
                    "python3",
                    os.path.join(PROJECT_DIR, "harvest_coordinator.py"),
                    "--robot-id",
                    robot_id,
                ],
                name=f"harvest_coordinator_{robot_id}",
                additional_env=env,
                output="screen",
                condition=UnlessCondition(execute),
            )
        )

    return LaunchDescription(actions)
