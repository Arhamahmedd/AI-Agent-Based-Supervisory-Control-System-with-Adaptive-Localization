#!/usr/bin/env python3
"""
Safety Agent — Day 7
Monitors /localization_health and publishes GREEN/AMBER/RED status.
"""

import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class SafetyAgent(Node):

    # Thresholds derived from Paper 1 baseline values
    HEALTH_AMBER  = 1.0     # health score below this -> AMBER (demo-tuned)
    HEALTH_RED    = 0.0     # health score below this -> RED (keeps hard floor)
    YAW_AMBER     = 0.20    # rad -> elevated concern (demo-tuned)
    YAW_RED       = 0.30    # rad -> critical (demo-tuned threshold)

    def __init__(self):
        super().__init__('safety_agent')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )])

        self.create_subscription(
            Float32MultiArray,
            '/localization_health',
            self.health_cb, 10)

        self.status_pub = self.create_publisher(
            String, '/supervisor/safety_status', 10)

        self.get_logger().info("Safety Agent running.")

    def health_cb(self, msg):
        if len(msg.data) < 4:
            return

        health    = float(msg.data[0])
        error_yaw = abs(float(msg.data[3]))

        if health < self.HEALTH_RED or error_yaw > self.YAW_RED:
            status = 'RED'
            reason = (f"Critical localization failure — "
                      f"health={health:.1f}, yaw_error={error_yaw:.4f} rad")
        elif health < self.HEALTH_AMBER or error_yaw > self.YAW_AMBER:
            status = 'AMBER'
            reason = (f"Elevated localization uncertainty — "
                      f"health={health:.1f}, yaw_error={error_yaw:.4f} rad")
        else:
            status = 'GREEN'
            reason = f"Localization nominal — health={health:.1f}"

        payload = {
            'status':       status,
            'reason':       reason,
            'health_score': round(health, 2),
            'yaw_error':    round(error_yaw, 4),
            'timestamp':    datetime.now().isoformat(),
        }

        out = String()
        out.data = json.dumps(payload)
        self.status_pub.publish(out)

        if status == 'RED':
            self.get_logger().error(f"[SAFETY RED] {reason}")
        elif status == 'AMBER':
            self.get_logger().warn(f"[SAFETY AMBER] {reason}",
                                   throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
