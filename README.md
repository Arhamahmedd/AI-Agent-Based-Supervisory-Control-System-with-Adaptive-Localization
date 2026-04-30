# AI Agent-Based Supervisory Control System with Adaptive Localization for Industrial Mobile Robots

A ROS 2 framework that combines an **LSTM-based adaptive localizer** with a **multi-agent supervisory control system** for industrial mobile robots. The system predicts localization errors in real time, dynamically updates EKF covariances, and uses a hierarchy of AI agents to make autonomous safety decisions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Packages](#packages)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the LSTM Model](#training-the-lstm-model)
- [Running the System](#running-the-system)
- [Web Dashboard](#web-dashboard)
- [Repository Structure](#repository-structure)
- [ROS 2 Topics](#ros-2-topics)
- [Configuration](#configuration)

---

## Overview

Industrial mobile robots rely on accurate localization to navigate safely. Traditional EKF-based approaches use fixed covariance matrices that cannot adapt to changing conditions (tight turns, slippery floors, sensor noise). This project addresses that limitation with two complementary components:

1. **LSTM Localizer** — A two-layer LSTM network trained on sliding windows of sensor data predicts positional and angular errors (`error_x`, `error_y`, `error_yaw`) at 5 Hz. The predicted errors are converted into dynamic covariances that are fed back into the EKF at runtime.

2. **AI Supervisor** — Three specialised ROS 2 agents (Safety, Flow, Override) monitor the robot's localization health and operator behaviour. A central Decision Engine aggregates their reports and issues actionable commands (`CONTINUE`, `REDUCE_SPEED`, `STOP`, `FLAG_MAINTENANCE`). Every decision is persisted to a SQLite audit log.

A **Flask web dashboard** provides a live view of all supervisor topics and the localization health history.

---

## Architecture

```
Gazebo Simulation (TurtleBot3)
          │
          ▼
  ┌───────────────────────────────────────────────┐
  │  Sensor Topics                                │
  │  /odom  /imu  /cmd_vel  /odometry/filtered    │
  └───────────────┬───────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  LSTM Covariance    │  ← lstm_covariance_node
        │  Node (5 Hz)        │    Predicts error_x/y/yaw
        └──────┬──────────────┘
               │  /ai_tools/covariance_prediction
               │  /localization_health
               ▼
   ┌───────────────────────┐        ┌──────────────────────┐
   │  AI Covariance        │        │  Supervisory Agents  │
   │  Updater              │        │                      │
   │  (EKF parameter sync) │        │  Safety Agent        │
   └───────────────────────┘        │  Flow Agent          │
                                    │  Override Audit Agent│
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  Decision Engine    │
                                    │  (1 Hz, SQLite log) │
                                    └──────────┬──────────┘
                                               │ /supervisor/decision
                                               ▼
                                    ┌─────────────────────┐
                                    │  Flask Dashboard    │
                                    │  localhost:5000     │
                                    └─────────────────────┘
```

---

## Packages

### `lstm_localizer`

| Node | Description |
|------|-------------|
| `lstm_covariance_node` | Runs LSTM inference every 0.2 s. Publishes predicted errors and a composite **health score** (0–100). |
| `ai_covariance_updater` | Converts predicted errors to covariances and updates the EKF via `set_parameters` service. |

### `ai_supervisor`

| Node | Description |
|------|-------------|
| `safety_agent` | Publishes **GREEN / AMBER / RED** based on health score and yaw error thresholds. |
| `flow_agent` | Monitors a rolling window (50 samples) of health scores; publishes **NOMINAL / DEGRADED**. |
| `override_agent` | Detects abrupt changes in `/cmd_vel` angular velocity as manual override events. |
| `decision_engine` | Aggregates all three agents every 1 s and issues a final decision. Logs to SQLite. |

---

## Prerequisites

| Dependency | Version |
|------------|---------|
| Ubuntu | 22.04 LTS |
| ROS 2 | Humble |
| TurtleBot3 packages | `turtlebot3_gazebo` |
| Python | 3.10+ |
| PyTorch | 1.13+ |
| scikit-learn | 1.x |
| Flask | 2.x |
| `robot_localization` | (EKF node) |

Install Python dependencies:

```bash
pip install torch numpy pandas scikit-learn joblib matplotlib flask
```

---

## Installation

```bash
# 1. Clone the repository into your ROS 2 workspace
mkdir -p ~/supervisor_ws/src && cd ~/supervisor_ws/src
git clone <repo-url> .

# 2. Build
cd ~/supervisor_ws
colcon build --symlink-install

# 3. Source the workspace
source ~/supervisor_ws/install/setup.bash
```

---

## Training the LSTM Model

Before running the system, a trained model must be available at `~/models/lstm_best.pt`.

### 1. Record / prepare bag data

```bash
# Process raw ROS bags into a structured dataset
python3 scripts/process_bags.py

# Prepare train/test splits and feature scaling
python3 scripts/prepare_data.py
```

### 2. Train the LSTM

```bash
python3 scripts/train_lstm.py
```

Training runs for 60 epochs. The best checkpoint (lowest validation loss) is saved to `~/models/lstm_best.pt`. Training curves are saved to `~/plots/lstm_training_curves.png`.

### 3. (Optional) Compare with baseline

```bash
python3 scripts/compare_models.py
```

---

## Running the System

### Full system (with Gazebo GUI)

```bash
ros2 launch ai_supervisor full_system.launch.py
```

### Headless (no GUI — faster on servers)

```bash
ros2 launch ai_supervisor full_system_headless.launch.py
```

The launch sequence is staggered automatically:

| Delay | Component |
|-------|-----------|
| 0 s | Gazebo + TurtleBot3 world |
| 5 s | EKF (`robot_localization`) |
| 8 s | LSTM Localizer |
| 12 s | Supervisory Agents + Decision Engine |

---

## Web Dashboard

Start the dashboard separately (it connects to the running ROS graph):

```bash
python3 ~/supervisor_ws/dashboard/app.py
```

Open **http://localhost:5000** in your browser.

| Endpoint | Description |
|----------|-------------|
| `/` | Live dashboard UI |
| `/api/state` | Full system state as JSON |
| `/api/decisions` | Last 100 decisions (live or from SQLite fallback) |
| `/api/health` | Localization health score history |
| `/api/status` | Diagnostic connection status |

When ROS topics are not flowing, the dashboard automatically falls back to reading the SQLite audit log at `~/logs/decisions.db`.

---

## Repository Structure

```
.
├── dashboard/
│   ├── app.py                  # Flask web dashboard
│   └── templates/
│       └── dashboard.html
├── launch/
│   ├── full_system.launch.py          # Full system launch (with GUI)
│   └── full_system_headless.launch.py # Headless launch
├── plots/
│   └── day7_validation_evidence.png
├── scripts/
│   ├── process_bags.py         # Extract features from ROS bags
│   ├── prepare_data.py         # Build train/test splits
│   ├── train_lstm.py           # Train the LSTM model
│   ├── train_baseline.py       # Train baseline comparison model
│   ├── compare_models.py       # Evaluate LSTM vs baseline
│   ├── validate_bags.py        # Validate bag integrity
│   └── validate_dataset.py     # Validate processed dataset
└── src/
    ├── ai_supervisor/
    │   └── ai_supervisor/
    │       ├── decision_engine.py   # Central decision logic + SQLite logging
    │       ├── flow_agent.py        # Rolling health trend monitor
    │       ├── override_agent.py    # Manual override detector
    │       └── safety_agent.py      # GREEN/AMBER/RED safety monitor
    └── lstm_localizer/
        ├── config/
        │   └── ekf.yaml             # EKF configuration
        └── lstm_localizer/
            ├── lstm_covariance_node.py    # LSTM inference node
            └── ai_covariance_updater.py   # EKF covariance updater
```

---

## ROS 2 Topics

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/localization_health` | `Float32MultiArray` | `lstm_covariance_node` | `[health, error_x, error_y, error_yaw]` |
| `/ai_tools/covariance_prediction` | `Float32MultiArray` | `lstm_covariance_node` | `[error_x, error_y, error_yaw]` |
| `/ai_tools/covariance` | `Float32MultiArray` | `ai_covariance_updater` | `[cov_x, cov_y, cov_yaw]` |
| `/supervisor/safety_status` | `String` (JSON) | `safety_agent` | GREEN / AMBER / RED + reason |
| `/supervisor/flow_status` | `String` (JSON) | `flow_agent` | NOMINAL / DEGRADED + stats |
| `/supervisor/override_log` | `String` (JSON) | `override_agent` | Override count + delta |
| `/supervisor/decision` | `String` (JSON) | `decision_engine` | CONTINUE / REDUCE_SPEED / STOP / FLAG_MAINTENANCE |

---

## Configuration

### EKF (`src/lstm_localizer/config/ekf.yaml`)

Edit this file to adjust the initial covariance matrix and sensor fusion settings for `robot_localization`.

### Safety thresholds (`safety_agent.py`)

```python
HEALTH_AMBER = 1.0    # health score below this → AMBER
HEALTH_RED   = 0.0    # health score below this → RED
YAW_AMBER    = 0.20   # rad yaw error → AMBER
YAW_RED      = 0.30   # rad yaw error → RED
```

### Flow agent (`flow_agent.py`)

```python
HISTORY_SIZE    = 50    # rolling window length
DEGRADED_THRESH = 50.0  # average health below this → DEGRADED
```

### Decision engine (`decision_engine.py`)

```python
EVAL_PERIOD = 1.0  # decision frequency (seconds)
```

A `STOP` command is issued only when `safety_status == RED` **and** `health_score < 2.0`. Otherwise a `REDUCE_SPEED` command is issued, allowing the robot to continue cautiously rather than halting unnecessarily.
