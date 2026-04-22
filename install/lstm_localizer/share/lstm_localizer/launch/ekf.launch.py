from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Find where your config file lives after building
    config_file = os.path.join(
        get_package_share_directory('lstm_localizer'),
        'config', 'ekf.yaml'
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',           # print all logs to terminal
        parameters=[config_file, {"use_sim_time": True}],  # load your yaml config
        remappings=[
            # Make sure EKF publishes to this exact topic name
            # Your LSTM node and supervisory agents will subscribe here
            ('odometry/filtered', '/odometry/filtered')
        ]
    )

    return LaunchDescription([ekf_node])
