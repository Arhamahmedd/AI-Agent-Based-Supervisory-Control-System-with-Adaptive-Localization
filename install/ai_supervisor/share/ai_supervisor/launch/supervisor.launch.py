from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='ai_supervisor',
            executable='safety_agent',
            name='safety_agent',
            output='screen',
        ),

        Node(
            package='ai_supervisor',
            executable='flow_agent',
            name='flow_agent',
            output='screen',
        ),

        Node(
            package='ai_supervisor',
            executable='override_agent',
            name='override_agent',
            output='screen',
        ),

        Node(
            package='ai_supervisor',
            executable='decision_engine',
            name='decision_engine',
            output='screen',
        ),

    ])
