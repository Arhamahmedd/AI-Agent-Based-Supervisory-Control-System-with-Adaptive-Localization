#!/usr/bin/env python3
"""
Day 4 — Bag Processing Pipeline
Converts MCAP ROS 2 bag files into LSTM-ready CSV dataset.
Project: AI-Agent Supervisory Control + LSTM Adaptive Localization
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages
import math
from datetime import datetime, timezone


# ── Configuration ────────────────────────────────────────────────────────────

BAG_PATHS = {
    'static':     os.path.expanduser('~/bags/static_run'),
    'moderate':   os.path.expanduser('~/bags/moderate_run'),
    'aggressive': os.path.expanduser('~/bags/aggressive_run'),
}

OUTPUT_DIR = os.path.expanduser('~/data')
COMBINED_CSV = os.path.join(OUTPUT_DIR, 'combined_dataset.csv')
SYNC_TOLERANCE_NS = 50_000_000   # 50ms — max time gap for message sync
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper: quaternion to yaw ─────────────────────────────────────────────────

def quat_to_yaw(x, y, z, w):
    """
    Convert quaternion orientation to yaw angle (rotation around Z axis).
    This is the heading angle — what direction the robot faces.
    Formula: yaw = 2 * arctan2(z, w)  (simplified for ground robots)
    """
    return 2.0 * math.atan2(z, w)


def to_nanoseconds(timestamp):
    """Convert a ROS bag timestamp to integer nanoseconds."""
    if hasattr(timestamp, 'value'):
        return int(timestamp.value)
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp() * 1_000_000_000)
    if hasattr(timestamp, 'total_seconds'):
        return int(timestamp.total_seconds() * 1_000_000_000)
    return int(timestamp)


# ── Step 1: Extract raw messages from one bag ─────────────────────────────────

def extract_messages(bag_path):
    """
    Read all messages from the four required topics.
    Returns four lists of (timestamp_ns, data_dict) tuples.
    """
    print(f"  Reading: {bag_path}")

    odom_msgs     = []
    filtered_msgs = []
    cmd_msgs      = []
    imu_msgs      = []

    # Find the .mcap file inside the bag directory
    mcap_files = list(Path(bag_path).glob('*.mcap'))
    if not mcap_files:
        raise FileNotFoundError(f"No MCAP file found in {bag_path}")

    mcap_file = str(mcap_files[0])
    print(f"  MCAP file: {mcap_file}")

    with open(mcap_file, 'rb') as f:
        for msg in read_ros2_messages(f):
            channel = msg.channel
            ros_msg = msg.ros_msg
            message = msg

            ts = to_nanoseconds(message.log_time)

            if channel.topic == '/odom':
                p = ros_msg.pose.pose
                t = ros_msg.twist.twist
                odom_msgs.append({
                    'ts':        ts,
                    'pos_x':     p.position.x,
                    'pos_y':     p.position.y,
                    'yaw_z':     p.orientation.z,
                    'yaw_w':     p.orientation.w,
                    'linear_x':  t.linear.x,
                    'angular_z': t.angular.z,
                })

            elif channel.topic == '/odometry/filtered':
                p = ros_msg.pose.pose
                t = ros_msg.twist.twist
                filtered_msgs.append({
                    'ts':            ts,
                    'pos_x_filt':    p.position.x,
                    'pos_y_filt':    p.position.y,
                    'yaw_z_filt':    p.orientation.z,
                    'yaw_w_filt':    p.orientation.w,
                    'linear_x_filt': t.linear.x,
                })

            elif channel.topic == '/cmd_vel':
                cmd_msgs.append({
                    'ts':             ts,
                    'cmd_linear_x':   ros_msg.twist.linear.x,
                    'cmd_angular_z':  ros_msg.twist.angular.z,
                })

            elif channel.topic == '/imu':
                la = ros_msg.linear_acceleration
                av = ros_msg.angular_velocity
                imu_msgs.append({
                    'ts':     ts,
                    'acc_x':  la.x,
                    'acc_y':  la.y,
                    'acc_z':  la.z,
                    'gyro_z': av.z,
                })

    print(f"  Messages — odom:{len(odom_msgs)} "
          f"filtered:{len(filtered_msgs)} "
          f"cmd:{len(cmd_msgs)} "
          f"imu:{len(imu_msgs)}")

    if len(filtered_msgs) == 0:
        raise ValueError(
            f"CRITICAL: /odometry/filtered is EMPTY in {bag_path}\n"
            "This means EKF was not running during recording.\n"
            "Re-record this bag with EKF confirmed running.")

    return odom_msgs, filtered_msgs, cmd_msgs, imu_msgs


# ── Step 2: Synchronise to common timestamps ──────────────────────────────────

def synchronise(odom_msgs, filtered_msgs, cmd_msgs, imu_msgs):
    """
    All four topics publish at different rates.
    We synchronise by matching each /odom message to the nearest
    message in each other topic within a 50ms tolerance window.

    Why /odom as reference: it publishes at ~30Hz which is the
    lowest common rate. EKF publishes faster (50Hz), IMU faster (40Hz).
    Using /odom as anchor avoids upsampling.
    """
    print("  Synchronising messages to /odom timestamps...")

    df_odom     = pd.DataFrame(odom_msgs).set_index('ts').sort_index()
    df_filtered = pd.DataFrame(filtered_msgs).set_index('ts').sort_index()
    df_cmd      = pd.DataFrame(cmd_msgs).set_index('ts').sort_index()
    df_imu      = pd.DataFrame(imu_msgs).set_index('ts').sort_index()

    rows = []

    for ts, odom_row in df_odom.iterrows():

        # Find nearest message in each topic within tolerance
        def nearest(df, timestamp):
            idx = df.index.searchsorted(timestamp)
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(df):
                candidates.append(idx)
            if not candidates:
                return None
            best = min(candidates, key=lambda i: abs(df.index[i] - timestamp))
            if abs(df.index[best] - timestamp) > SYNC_TOLERANCE_NS:
                return None
            return df.iloc[best]

        filt = nearest(df_filtered, ts)
        cmd  = nearest(df_cmd, ts)
        imu  = nearest(df_imu, ts)

        # Only keep rows where all four sources matched
        if filt is None or cmd is None or imu is None:
            continue

        rows.append({
            'timestamp_ns':   ts,
            # Raw odometry
            'pos_x_odom':     odom_row['pos_x'],
            'pos_y_odom':     odom_row['pos_y'],
            'linear_x':       odom_row['linear_x'],
            'angular_z':      odom_row['angular_z'],
            'yaw_z_odom':     odom_row['yaw_z'],
            'yaw_w_odom':     odom_row['yaw_w'],
            # Filtered odometry
            'pos_x_filtered': filt['pos_x_filt'],
            'pos_y_filtered': filt['pos_y_filt'],
            'yaw_z_filtered': filt['yaw_z_filt'],
            'yaw_w_filtered': filt['yaw_w_filt'],
            # Commands
            'cmd_linear_x':   cmd['cmd_linear_x'],
            'cmd_angular_z':  cmd['cmd_angular_z'],
            # IMU
            'acc_x':          imu['acc_x'],
            'acc_y':          imu['acc_y'],
            'acc_z':          imu['acc_z'],
            'gyro_z':         imu['gyro_z'],
        })

    df = pd.DataFrame(rows).sort_values('timestamp_ns').reset_index(drop=True)
    print(f"  Synchronised rows: {len(df)}")
    return df


# ── Step 3: Engineer features and compute targets ─────────────────────────────

def engineer_features(df):
    """
    Compute all 15 features and 3 targets from raw synchronised data.
    """
    print("  Engineering features...")

    # Orientation angles from quaternion
    # arctan2(z, w) * 2 gives yaw in radians [-π, π]
    df['yaw_odom']     = df.apply(
        lambda r: quat_to_yaw(0, 0, r['yaw_z_odom'], r['yaw_w_odom']), axis=1)
    df['yaw_filtered'] = df.apply(
        lambda r: quat_to_yaw(0, 0, r['yaw_z_filtered'], r['yaw_w_filtered']), axis=1)

    # Yaw discrepancy — how much odom and EKF disagree on orientation
    # This is itself a feature AND related to error_yaw target
    df['yaw_diff'] = df['yaw_odom'] - df['yaw_filtered']

    # Interaction terms — capture cross-sensor relationships
    # acc_y × gyro_z: lateral acceleration during turning
    # gyro_z × linear_x: centripetal effect during forward+turning
    df['acc_y_mul_gyro_z']    = df['acc_y'] * df['gyro_z']
    df['gyro_z_mul_linear_x'] = df['gyro_z'] * df['linear_x']

    # Temporal deltas — rate of change (1st derivative)
    # These capture acceleration in angular motion — critical for
    # detecting the ONSET of aggressive manoeuvres
    df['delta_angular_z']     = df['angular_z'].diff().fillna(0.0)
    df['delta_cmd_angular_z'] = df['cmd_angular_z'].diff().fillna(0.0)

    # ── Training targets (what LSTM predicts) ────────────────────────
    df['error_x']   = df['pos_x_odom']  - df['pos_x_filtered']
    df['error_y']   = df['pos_y_odom']  - df['pos_y_filtered']
    df['error_yaw'] = df['yaw_odom']    - df['yaw_filtered']

    return df


# ── Step 4: Select final columns ──────────────────────────────────────────────

FEATURES = [
    'acc_x', 'acc_y', 'acc_z',
    'gyro_z',
    'linear_x', 'angular_z',
    'yaw_odom', 'yaw_filtered', 'yaw_diff',
    'acc_y_mul_gyro_z', 'gyro_z_mul_linear_x',
    'cmd_linear_x', 'cmd_angular_z',
    'delta_angular_z', 'delta_cmd_angular_z',
]

TARGETS = ['error_x', 'error_y', 'error_yaw']

META = ['timestamp_ns', 'profile']


def select_columns(df, profile_name):
    df['profile'] = profile_name
    keep = META + FEATURES + TARGETS
    df = df[keep].dropna()
    return df


# ── Step 5: Remove outliers ───────────────────────────────────────────────────

def remove_outliers(df):
    """
    Remove rows with extreme error values that are physically impossible.
    These come from bag start/stop transients or EKF initialisation.
    Cap at 3 radians for yaw (almost a full turn) and 2 metres for position.
    """
    original_len = len(df)

    df = df[df['error_yaw'].abs() < 3.0]
    df = df[df['error_x'].abs()   < 2.0]
    df = df[df['error_y'].abs()   < 2.0]

    removed = original_len - len(df)
    if removed > 0:
        print(f"  Removed {removed} outlier rows ({removed/original_len*100:.1f}%)")

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("\nDay 4 — Bag Processing Pipeline")
    print("="*50)

    all_dfs = []

    for profile_name, bag_path in BAG_PATHS.items():
        print(f"\nProcessing: {profile_name.upper()}")

        if not os.path.exists(bag_path):
            print(f"  SKIP: Path not found: {bag_path}")
            continue

        # Extract
        odom, filtered, cmd, imu = extract_messages(bag_path)

        # Synchronise
        df_sync = synchronise(odom, filtered, cmd, imu)

        # Feature engineering
        df_feat = engineer_features(df_sync)

        # Select columns
        df_final = select_columns(df_feat, profile_name)

        # Remove outliers
        df_final = remove_outliers(df_final)

        print(f"  Final rows for {profile_name}: {len(df_final)}")
        all_dfs.append(df_final)

    if not all_dfs:
        print("\nERROR: No bags were processed. Check paths.")
        return

    # Combine all profiles
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined)} rows")

    # Profile distribution
    print("\nProfile distribution:")
    for profile, count in combined['profile'].value_counts().items():
        pct = count / len(combined) * 100
        print(f"  {profile:12s}: {count:6d} rows ({pct:.1f}%)")

    # Feature statistics
    print("\nTarget statistics:")
    for col in TARGETS:
        s = combined[col]
        print(f"  {col:12s}: mean={s.mean():.5f}  "
              f"std={s.std():.5f}  "
              f"max_abs={s.abs().max():.5f}")

    # Save
    combined.to_csv(COMBINED_CSV, index=False)
    print(f"\nSaved to: {COMBINED_CSV}")
    print(f"File size: {os.path.getsize(COMBINED_CSV) / 1024 / 1024:.1f} MB")

    # Final shape confirmation
    print(f"\nDataset shape: {combined.shape}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"Targets  ({len(TARGETS)}):  {TARGETS}")
    print(f"\nDay 4 complete.")


if __name__ == '__main__':
    main()