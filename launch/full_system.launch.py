"""
Master Launch File — Day 8
Starts: Gazebo → EKF → LSTM Localizer → Supervisory Agents
Run dashboard separately: python3 ~/supervisor_ws/dashboard/app.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'launch', 'turtlebot3_world.launch.py'
            )
        )
    )

    ekf_launch = TimerAction(
        period=5.0,  # wait 5s for Gazebo to fully load
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('lstm_localizer'),
                    'launch', 'ekf.launch.py'
                )
            )
        )]
    )

    lstm_launch = TimerAction(
        period=8.0,  # wait for EKF to initialise
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('lstm_localizer'),
                    'launch', 'lstm_localizer.launch.py'
                )
            )
        )]
    )

    supervisor_launch = TimerAction(
        period=12.0,  # wait for LSTM topics to appear
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('ai_supervisor'),
                    'launch', 'supervisor.launch.py'
                )
            )
        )]
    )

    return LaunchDescription([
        gazebo_launch,
        ekf_launch,
        lstm_launch,
        supervisor_launch,
    ])
