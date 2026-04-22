#!/usr/bin/env python3
"""
LSTM Covariance Node — Day 6
Reads sensor topics, runs LSTM inference every 0.2s,
publishes covariance predictions and localization health score.
"""

import os
import csv
import math
import time
import torch
import joblib
import numpy as np
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped


# ── Model definition (must match train_lstm.py exactly) ──────────────────────

class LSTMPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=15, hidden_size=64,
            num_layers=2, dropout=0.2, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 3))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Main node ─────────────────────────────────────────────────────────────────

class LSTMCovarianceNode(Node):

    WINDOW_SIZE = 20
    INFER_PERIOD = 0.2   # seconds between inference calls (5 Hz)
    MODELS_DIR = os.path.expanduser('~/models')
    LOGS_DIR   = os.path.expanduser('~/logs')

    def __init__(self):
        super().__init__('lstm_covariance_node')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )])

        os.makedirs(self.LOGS_DIR, exist_ok=True)

        # ── Load model and scalers
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        self.model = LSTMPredictor().to(self.device)
        model_path = os.path.join(self.MODELS_DIR, 'lstm_best.pt')
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.scaler_X = joblib.load(
            os.path.join(self.MODELS_DIR, 'scaler_X.joblib'))
        self.scaler_y = joblib.load(
            os.path.join(self.MODELS_DIR, 'scaler_y.joblib'))

        self.get_logger().info("Model and scalers loaded.")

        # ── Sliding window buffer
        self.buffer = deque(maxlen=self.WINDOW_SIZE)

        # ── Latest sensor readings
        self.odom_data     = None
        self.imu_data      = None
        self.cmd_data      = None
        self.filtered_data = None

        # ── For delta features
        self.prev_angular_z     = 0.0
        self.prev_cmd_angular_z = 0.0

        # ── Subscriptions
        self.create_subscription(
            Odometry, '/odom',
            self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(
            Imu, '/imu',
            self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(
            TwistStamped, '/cmd_vel',
            self.cmd_stamped_cb, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, '/odometry/filtered',
            self.filtered_cb, qos_profile_sensor_data)

        # ── Publishers
        self.pred_pub = self.create_publisher(
            Float32MultiArray, '/ai_tools/covariance_prediction', 10)
        self.health_pub = self.create_publisher(
            Float32MultiArray, '/localization_health', 10)

        # ── Inference timer
        self.create_timer(self.INFER_PERIOD, self.run_inference)

        # ── CSV logger
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(self.LOGS_DIR, f'lstm_{ts}.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp_ns', 'error_x', 'error_y', 'error_yaw',
            'health_score', 'latency_ms'
        ])

        self.get_logger().info("LSTM Covariance Node ready. Waiting for topics...")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def odom_cb(self, msg):
        self.odom_data = msg

    def imu_cb(self, msg):
        self.imu_data = msg

    def cmd_stamped_cb(self, msg):
        self.cmd_data = msg.twist

    def filtered_cb(self, msg):
        self.filtered_data = msg

    # ── Feature builder ───────────────────────────────────────────────────────

    @staticmethod
    def quat_to_yaw(z, w):
        return 2.0 * math.atan2(z, w)

    def build_feature_vector(self):
        if any(x is None for x in [
                self.odom_data, self.imu_data,
                self.cmd_data, self.filtered_data]):
            return None

        # Odometry
        linear_x  = self.odom_data.twist.twist.linear.x
        angular_z = self.odom_data.twist.twist.angular.z
        q_odom    = self.odom_data.pose.pose.orientation
        yaw_odom  = self.quat_to_yaw(q_odom.z, q_odom.w)

        # Filtered odometry
        q_filt      = self.filtered_data.pose.pose.orientation
        yaw_filtered = self.quat_to_yaw(q_filt.z, q_filt.w)
        yaw_diff     = yaw_odom - yaw_filtered

        # IMU
        acc_x  = self.imu_data.linear_acceleration.x
        acc_y  = self.imu_data.linear_acceleration.y
        acc_z  = self.imu_data.linear_acceleration.z
        gyro_z = self.imu_data.angular_velocity.z

        # Commands
        cmd_linear_x  = self.cmd_data.linear.x
        cmd_angular_z = self.cmd_data.angular.z

        # Temporal deltas
        delta_angular_z     = angular_z     - self.prev_angular_z
        delta_cmd_angular_z = cmd_angular_z - self.prev_cmd_angular_z
        self.prev_angular_z     = angular_z
        self.prev_cmd_angular_z = cmd_angular_z

        return [
            acc_x, acc_y, acc_z,
            gyro_z,
            linear_x, angular_z,
            yaw_odom, yaw_filtered, yaw_diff,
            acc_y * gyro_z,
            gyro_z * linear_x,
            cmd_linear_x, cmd_angular_z,
            delta_angular_z, delta_cmd_angular_z,
        ]

    # ── Inference ─────────────────────────────────────────────────────────────

    def run_inference(self):
        features = self.build_feature_vector()
        if features is None:
            self.get_logger().warn(
                "Waiting for all topics: /odom /imu /cmd_vel /odometry/filtered",
                throttle_duration_sec=5.0)
            return

        self.buffer.append(features)

        if len(self.buffer) < self.WINDOW_SIZE:
            return

        t0 = time.perf_counter()

        # Build scaled window tensor
        window_raw = np.array(list(self.buffer), dtype=np.float32)  # [20, 15]
        window_scaled = self.scaler_X.transform(window_raw)          # [20, 15]
        x_tensor = torch.FloatTensor(window_scaled).unsqueeze(0).to(self.device)
        # x_tensor shape: [1, 20, 15]

        with torch.no_grad():
            pred_scaled = self.model(x_tensor).cpu().numpy()  # [1, 3]

        pred = self.scaler_y.inverse_transform(pred_scaled)[0]  # [3]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        error_x, error_y, error_yaw = float(pred[0]), float(pred[1]), float(pred[2])

        # ── Publish covariance prediction
        cov_msg = Float32MultiArray()
        cov_msg.data = [error_x, error_y, error_yaw]
        self.pred_pub.publish(cov_msg)

        # ── Compute and publish health score
        # health = 100 when error_yaw ≈ 0, drops as error grows
        # 0.15 rad error → health = 0 (maps Paper 1 aggressive threshold)
        health = float(max(0.0, 100.0 - (abs(error_yaw) / 0.0015)))
        health = min(100.0, health)

        health_msg = Float32MultiArray()
        health_msg.data = [health, error_x, error_y, error_yaw]
        self.health_pub.publish(health_msg)

        # ── Log to CSV
        ts_ns = self.get_clock().now().nanoseconds
        self.csv_writer.writerow([
            ts_ns, error_x, error_y, error_yaw, health, round(latency_ms, 2)
        ])

        self.get_logger().info(
            f"Δx={error_x:.4f}m  Δy={error_y:.4f}m  "
            f"Δyaw={error_yaw:.4f}rad  "
            f"health={health:.1f}  latency={latency_ms:.1f}ms",
            throttle_duration_sec=1.0)

    def destroy_node(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LSTMCovarianceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
