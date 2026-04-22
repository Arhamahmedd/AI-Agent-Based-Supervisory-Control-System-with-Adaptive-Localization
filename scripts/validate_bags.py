#!/usr/bin/env python3
"""
Quick validation script for recorded bag files.
Run this before processing to catch problems early.
"""
import os
import subprocess
import sys

def check_bag(bag_path, profile_name):
    print(f"\n{'='*50}")
    print(f"Checking: {profile_name}")
    print(f"Path: {bag_path}")
    print('='*50)
    
    result = subprocess.run(
        ['ros2', 'bag', 'info', bag_path],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: Could not read bag file")
        print(result.stderr)
        return False
    
    output = result.stdout
    print(output)
    
    # Check all required topics exist
    required_topics = ['/odom', '/odometry/filtered', '/cmd_vel', '/imu']
    missing = []
    for topic in required_topics:
        if topic not in output:
            missing.append(topic)
    
    if missing:
        print(f"\nCRITICAL: Missing topics: {missing}")
        print("This bag CANNOT be used for training. Re-record with EKF running.")
        return False
    
    # Check filtered odometry has messages
    if 'Count: 0' in output and '/odometry/filtered' in output:
        print("\nCRITICAL: /odometry/filtered has 0 messages")
        print("EKF was not running during recording. Re-record.")
        return False
    
    print("\nBag validation: PASSED")
    return True

# Check all three bags
bags = [
    (os.path.expanduser('~/bags/static_run'), 'Static Profile'),
    (os.path.expanduser('~/bags/moderate_run'), 'Moderate Profile'),
    (os.path.expanduser('~/bags/aggressive_run'), 'Aggressive Profile'),
]

all_ok = True
for path, name in bags:
    ok = check_bag(path, name)
    if not ok:
        all_ok = False

print(f"\n{'='*50}")
if all_ok:
    print("ALL BAGS VALID — Ready to process on Day 4")
else:
    print("SOME BAGS FAILED — Fix issues before Day 4")
print('='*50)
