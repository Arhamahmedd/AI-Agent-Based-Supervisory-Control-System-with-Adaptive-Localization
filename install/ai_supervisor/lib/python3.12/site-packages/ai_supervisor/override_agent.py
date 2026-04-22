#!/usr/bin/env python3
"""
Override Audit Agent — Day 7
Detects sudden large changes in /cmd_vel angular velocity.
These indicate manual override or aggressive operator intervention.
"""

import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class OverrideAuditAgent(Node):

    # A delta > 0.5 rad/s in angular_z between consecutive commands
    # is classified as a manual override event
    OVERRIDE_THRESHOLD = 0.5

    def __init__(self):
        super().__init__('override_agent')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )])

        self.prev_angular_z = 0.0
        self.override_count = 0

        self.create_subscription(
            Twist, '/cmd_vel', self.cmd_cb, 10)

        self.log_pub = self.create_publisher(
            String, '/supervisor/override_log', 10)

        self.get_logger().info("Override Audit Agent running.")

    def cmd_cb(self, msg):
        current_az = msg.angular.z
        delta = abs(current_az - self.prev_angular_z)

        if delta > self.OVERRIDE_THRESHOLD:
            self.override_count += 1

            payload = {
                'override_number': self.override_count,
                'delta_angular_z': round(delta, 4),
                'current_angular_z': round(current_az, 4),
                'timestamp': datetime.now().isoformat(),
            }

            out = String()
            out.data = json.dumps(payload)
            self.log_pub.publish(out)

            self.get_logger().warn(
                f"Override #{self.override_count} — "
                f"delta_angular_z={delta:.4f} rad/s")

        self.prev_angular_z = current_az


def main(args=None):
    rclpy.init(args=args)
    node = OverrideAuditAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
