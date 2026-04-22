#!/bin/bash
# Usage: ./record_result_bag.sh <profile> <mode>
# profile: static | moderate | aggressive
# mode: lstm | baseline
#
# Example: ./record_result_bag.sh aggressive lstm

PROFILE=$1
MODE=$2
SOURCE_BAG=~/bags/${PROFILE}_run
OUTPUT_BAG=~/bags/results/${PROFILE}_${MODE}

if [ -z "$PROFILE" ] || [ -z "$MODE" ]; then
    echo "Usage: $0 <static|moderate|aggressive> <lstm|baseline>"
    exit 1
fi

if [ ! -d "$SOURCE_BAG" ]; then
    echo "ERROR: Source bag not found: $SOURCE_BAG"
    exit 1
fi

mkdir -p ~/bags/results
echo ""
echo "============================================"
echo "Profile : $PROFILE"
echo "Mode    : $MODE"
echo "Source  : $SOURCE_BAG"
echo "Output  : $OUTPUT_BAG"
echo "============================================"
echo ""

if [ "$MODE" = "lstm" ]; then
    TOPICS="/odom /odometry/filtered /cmd_vel /imu \
            /ai_tools/covariance_prediction /localization_health \
            /supervisor/decision /supervisor/safety_status"
else
    TOPICS="/odom /odometry/filtered /cmd_vel /imu"
fi

# Start recorder
ros2 bag record -o $OUTPUT_BAG $TOPICS &
RECORD_PID=$!
echo "Recorder started (PID $RECORD_PID)"
sleep 2

# Play source bag
echo "Replaying bag at 1x speed..."
ros2 bag play $SOURCE_BAG \
    --clock \
    --rate 1.0 \
    --topics /odom /odometry/filtered /cmd_vel /imu

echo "Replay finished. Stopping recorder..."
sleep 1
kill $RECORD_PID
wait $RECORD_PID 2>/dev/null

echo ""
echo "Result bag recorded: $OUTPUT_BAG"
ros2 bag info $OUTPUT_BAG | grep -E "Duration|Messages|Topic"
echo "============================================"
