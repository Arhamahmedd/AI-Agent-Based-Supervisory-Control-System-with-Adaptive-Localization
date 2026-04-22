#!/usr/bin/env python3
"""
Flow Agent — Day 7
Monitors sustained localization quality trend.
Publishes NOMINAL or DEGRADED status every 2 seconds.
"""

import json
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class FlowAgent(Node):

    EVAL_PERIOD     = 2.0   # evaluate every 2 seconds
    HISTORY_SIZE    = 50    # rolling window of health scores
    DEGRADED_THRESH = 50.0  # avg health below this -> DEGRADED

    def __init__(self):
        super().__init__('flow_agent')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )])

        self.health_history = deque(maxlen=self.HISTORY_SIZE)

        self.create_subscription(
            Float32MultiArray,
            '/localization_health',
            self.health_cb, 10)

        self.status_pub = self.create_publisher(
            String, '/supervisor/flow_status', 10)

        self.create_timer(self.EVAL_PERIOD, self.evaluate_flow)

        self.get_logger().info("Flow Agent running.")

    def health_cb(self, msg):
        if len(msg.data) >= 1:
            self.health_history.append(float(msg.data[0]))

    def evaluate_flow(self):
        if len(self.health_history) < 5:
            return

        avg_health = float(np.mean(list(self.health_history)))
        min_health = float(np.min(list(self.health_history)))

        if avg_health < self.DEGRADED_THRESH:
            flow_status    = 'DEGRADED'
            recommendation = (
                f"Navigation efficiency reduced. Avg health={avg_health:.1f}. "
                "Consider reducing speed or recalibrating sensors.")
        else:
            flow_status    = 'NOMINAL'
            recommendation = f"Navigation flow normal. Avg health={avg_health:.1f}."

        payload = {
            'status':         flow_status,
            'avg_health':     round(avg_health, 2),
            'min_health':     round(min_health, 2),
            'recommendation': recommendation,
            'timestamp':      datetime.now().isoformat(),
        }

        out = String()
        out.data = json.dumps(payload)
        self.status_pub.publish(out)

        if flow_status == 'DEGRADED':
            self.get_logger().warn(f"[FLOW DEGRADED] {recommendation}")


def main(args=None):
    rclpy.init(args=args)
    node = FlowAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
