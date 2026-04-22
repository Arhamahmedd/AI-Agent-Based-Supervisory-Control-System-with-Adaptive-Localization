#!/usr/bin/env bash
set -euo pipefail

# Replay an existing profile bag and record required result topics.
# Usage: ./scripts/replay_and_record.sh <profile>
# profile: static | moderate | aggressive

PROFILE="${1:-}"
if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 <profile>"
  exit 1
fi

SOURCE_BAG="$HOME/bags/${PROFILE}_run"
OUTPUT_BAG="$HOME/bags/results/${PROFILE}_lstm"

if [[ ! -d "$SOURCE_BAG" ]]; then
  echo "Source bag not found: $SOURCE_BAG"
  exit 2
fi

mkdir -p "$HOME/bags/results"

echo "=== Replaying $SOURCE_BAG and recording to $OUTPUT_BAG ==="

ros2 bag record \
  -o "$OUTPUT_BAG" \
  /odom /odometry/filtered /cmd_vel /imu \
  /ai_tools/covariance_prediction /localization_health \
  /supervisor/decision /supervisor/safety_status &
REC_PID=$!

cleanup() {
  if kill -0 "$REC_PID" 2>/dev/null; then
    kill "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

sleep 3
ros2 bag play "$SOURCE_BAG" --clock --rate 1.0
cleanup

echo "Done: $OUTPUT_BAG"
ros2 bag info "$OUTPUT_BAG" | grep -E 'Duration|Messages|Topic' || true
#!/usr/bin/env bash
set -euo pipefail

# Replay an input bag and simultaneously record evaluation topics.
# Usage:
#   ./scripts/replay_and_record.sh <profile> <mode>
# Where:
#   profile: static | moderate | aggressive
#   mode:    lstm | baseline

PROFILE="${1:-}"
MODE="${2:-}"

if [[ -z "$PROFILE" || -z "$MODE" ]]; then
  echo "Usage: $0 <profile> <mode>"
  echo "  profile: static|moderate|aggressive"
  echo "  mode:    lstm|baseline"
  exit 1
fi

if [[ "$PROFILE" != "static" && "$PROFILE" != "moderate" && "$PROFILE" != "aggressive" ]]; then
  echo "Invalid profile: $PROFILE"
  exit 1
fi

if [[ "$MODE" != "lstm" && "$MODE" != "baseline" ]]; then
  echo "Invalid mode: $MODE"
  exit 1
fi

SRC_BAG="$HOME/bags/${PROFILE}_run"
OUT_BAG="$HOME/bags/results/${PROFILE}_${MODE}"

if [[ ! -d "$SRC_BAG" ]]; then
  echo "Source bag not found: $SRC_BAG"
  echo "Expected folders like ~/bags/static_run"
  exit 1
fi

mkdir -p "$HOME/bags/results"

if [[ "$MODE" == "lstm" ]]; then
  TOPICS=(
    /odom
    /odometry/filtered
    /cmd_vel
    /imu
    /ai_tools/covariance_prediction
    /localization_health
    /supervisor/decision
    /supervisor/safety_status
  )
else
  TOPICS=(
    /odom
    /odometry/filtered
    /cmd_vel
    /imu
  )
fi

echo "Recording: $OUT_BAG"
echo "Replaying : $SRC_BAG"

ros2 bag record -o "$OUT_BAG" "${TOPICS[@]}" >/tmp/replay_record_${PROFILE}_${MODE}.log 2>&1 &
RECORD_PID=$!

# Give recorder time to subscribe before playback starts.
sleep 3

ros2 bag play "$SRC_BAG" --clock --rate 1.0 --disable-keyboard-controls

# Stop recorder gracefully if still running.
if kill -0 "$RECORD_PID" 2>/dev/null; then
  kill -INT "$RECORD_PID"
  wait "$RECORD_PID" || true
fi

echo "Done: $OUT_BAG"
ros2 bag info "$OUT_BAG" | sed -n '1,120p'
