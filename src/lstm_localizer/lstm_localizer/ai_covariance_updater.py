#!/usr/bin/env python3
"""
AI Covariance Updater — adapted from Abaza (2025) ai_tools
Subscribes to LSTM predictions, computes covariances, updates EKF via service.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters


class AICovarianceUpdater(Node):

    def __init__(self):
        super().__init__('ai_covariance_updater')

        self.create_subscription(
            Float32MultiArray,
            '/ai_tools/covariance_prediction',
            self.prediction_cb, 10)

        self.cov_pub = self.create_publisher(
            Float32MultiArray, '/ai_tools/covariance', 10)

        # Service client to update EKF parameters at runtime
        self.ekf_client = self.create_client(
            SetParameters, '/ekf_filter_node/set_parameters')

        self.get_logger().info("AI Covariance Updater ready.")

    def prediction_cb(self, msg):
        if len(msg.data) < 3:
            return

        error_x, error_y, error_yaw = msg.data[0], msg.data[1], msg.data[2]

        # Covariance = squared error (variance)
        cov_x   = float(error_x   ** 2)
        cov_y   = float(error_y   ** 2)
        cov_yaw = float(error_yaw ** 2)

        # Clamp to sensible minimum to avoid zero covariance
        cov_x   = max(cov_x,   1e-6)
        cov_y   = max(cov_y,   1e-6)
        cov_yaw = max(cov_yaw, 1e-6)

        # Publish covariance values
        cov_msg = Float32MultiArray()
        cov_msg.data = [cov_x, cov_y, cov_yaw]
        self.cov_pub.publish(cov_msg)

        # Build 6x6 covariance matrix (row-major, 36 elements)
        # positions 0, 7, 35 are x, y, yaw diagonal entries
        covariance_matrix = [1e-3] * 36
        covariance_matrix[0]  = cov_x
        covariance_matrix[7]  = cov_y
        covariance_matrix[35] = cov_yaw

        # Update EKF asynchronously
        if self.ekf_client.service_is_ready():
            request = SetParameters.Request()
            param_val = ParameterValue()
            param_val.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            param_val.double_array_value = covariance_matrix
            param = Parameter()
            param.name  = 'initial_estimate_covariance'
            param.value = param_val
            request.parameters = [param]
            self.ekf_client.call_async(request)

        self.get_logger().info(
            f"Covariances updated — x={cov_x:.6f}  "
            f"y={cov_y:.6f}  yaw={cov_yaw:.6f}",
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = AICovarianceUpdater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
