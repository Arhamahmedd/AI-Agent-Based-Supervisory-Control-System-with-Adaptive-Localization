from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='lstm_localizer',
            executable='lstm_covariance_node',
            name='lstm_covariance_node',
            output='screen',
        ),

        Node(
            package='lstm_localizer',
            executable='ai_covariance_updater',
            name='ai_covariance_updater',
            output='screen',
        ),

    ])
