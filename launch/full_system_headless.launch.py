"""
Headless Master Launch
Starts: Gazebo server only (no GUI client) -> EKF -> LSTM -> Supervisor
Use this when gz GUI crashes due to host runtime conflicts.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(tb3_gazebo, 'worlds', 'turtlebot3_world.world')

    # Server-only sim avoids launching the gz GUI process (-g).
    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r -s -v2 ', world],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    robot_state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    spawn_turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={'x_pose': '0.0', 'y_pose': '0.0'}.items(),
    )

    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gazebo, 'models'),
    )

    ekf_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('lstm_localizer'),
                        'launch',
                        'ekf.launch.py',
                    )
                )
            )
        ],
    )

    lstm_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('lstm_localizer'),
                        'launch',
                        'lstm_localizer.launch.py',
                    )
                )
            )
        ],
    )

    supervisor_launch = TimerAction(
        period=12.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('ai_supervisor'),
                        'launch',
                        'supervisor.launch.py',
                    )
                )
            )
        ],
    )

    return LaunchDescription(
        [
            set_resource_path,
            gzserver_launch,
            robot_state_publisher_launch,
            spawn_turtlebot_launch,
            ekf_launch,
            lstm_launch,
            supervisor_launch,
        ]
    )
"""
Headless Full System Launch
Starts: Gazebo server (no GUI) -> EKF -> LSTM Localizer -> Supervisory Agents
Use this when gzclient crashes due to host graphics/runtime library conflicts.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world = os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world')

    # Server-only Gazebo simulation. Intentionally omits gzclient (-g) process.
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items(),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_share, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_share, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={'x_pose': '-2.0', 'y_pose': '-0.5'}.items(),
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(tb3_gazebo_share, 'models'),
    )

    ekf_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('lstm_localizer'),
                        'launch',
                        'ekf.launch.py',
                    )
                )
            )
        ],
    )

    lstm_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('lstm_localizer'),
                        'launch',
                        'lstm_localizer.launch.py',
                    )
                )
            )
        ],
    )

    supervisor_launch = TimerAction(
        period=12.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('ai_supervisor'),
                        'launch',
                        'supervisor.launch.py',
                    )
                )
            )
        ],
    )

    return LaunchDescription(
        [
            set_gz_resource_path,
            gzserver_cmd,
            robot_state_publisher_cmd,
            spawn_turtlebot_cmd,
            ekf_launch,
            lstm_launch,
            supervisor_launch,
        ]
    )
