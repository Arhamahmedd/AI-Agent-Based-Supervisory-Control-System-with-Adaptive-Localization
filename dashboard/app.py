#!/usr/bin/env python3
"""
Supervisor Dashboard — Fixed Version
Key fixes:
  1. rclpy init happens inside the thread with error handling
  2. State updates are thread-safe
  3. Topics are subscribed with exact matching names
  4. API returns copy of state to avoid lock contention
  5. Fallback to SQLite if live topics not flowing
"""

import json
import threading
import os
import sqlite3
import time
import traceback

from flask import Flask, jsonify, render_template

# ── Shared state ──────────────────────────────────────────────────────────────

state_lock = threading.Lock()

state = {
    'safety': {
        'status':      'WAITING',
        'reason':      'Connecting to ROS...',
        'health_score': 0.0,
        'yaw_error':   0.0,
        'timestamp':   '',
    },
    'flow': {
        'status':         'WAITING',
        'avg_health':     0.0,
        'min_health':     0.0,
        'recommendation': 'Connecting to ROS...',
        'timestamp':      '',
    },
    'decision': {
        'decision':  'WAITING',
        'reason':    'Connecting to ROS...',
        'safety':    'WAITING',
        'flow':      'WAITING',
        'overrides': 0,
        'timestamp': '',
    },
    'health_history':   [],
    'decision_history': [],
    'override_count':   0,
    'ros_connected':    False,
    'last_update':      '',
}

ros_ok = False


def update_state(key, value):
    """Thread-safe state update."""
    with state_lock:
        state[key] = value
        state['last_update'] = time.strftime('%H:%M:%S')
        state['ros_connected'] = True


# ── ROS 2 listener ────────────────────────────────────────────────────────────

def ros_thread_fn():
    """
    Runs rclpy in a background thread.
    All imports happen inside the thread to avoid cross-thread rclpy issues.
    """
    global ros_ok
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String, Float32MultiArray

        rclpy.init()

        node = rclpy.create_node('dashboard_listener')
        print("ROS NODE STARTED")

        # ── Safety status callback
        def safety_cb(msg):
            try:
                print("RECEIVED SAFETY")
                d = json.loads(msg.data)
                update_state('safety', {
                    'status':       d.get('status', 'UNKNOWN'),
                    'reason':       d.get('reason', ''),
                    'health_score': float(d.get('health_score', 0.0)),
                    'yaw_error':    float(d.get('yaw_error', 0.0)),
                    'timestamp':    d.get('timestamp', ''),
                })
            except Exception as e:
                node.get_logger().warn(f"safety_cb error: {e}")

        # ── Flow status callback
        def flow_cb(msg):
            try:
                d = json.loads(msg.data)
                update_state('flow', {
                    'status':         d.get('status', 'UNKNOWN'),
                    'avg_health':     float(d.get('avg_health', 0.0)),
                    'min_health':     float(d.get('min_health', 0.0)),
                    'recommendation': d.get('recommendation', ''),
                    'timestamp':      d.get('timestamp', ''),
                })
            except Exception as e:
                node.get_logger().warn(f"flow_cb error: {e}")

        # ── Decision callback
        def decision_cb(msg):
            try:
                print("RECEIVED DECISION")
                d = json.loads(msg.data)
                update_state('decision', {
                    'decision':  d.get('decision', 'UNKNOWN'),
                    'reason':    d.get('reason', ''),
                    'safety':    d.get('safety', 'UNKNOWN'),
                    'flow':      d.get('flow', 'UNKNOWN'),
                    'overrides': int(d.get('overrides', 0)),
                    'timestamp': d.get('timestamp', ''),
                })
                with state_lock:
                    state['decision_history'].append(d)
                    if len(state['decision_history']) > 100:
                        state['decision_history'].pop(0)
            except Exception as e:
                node.get_logger().warn(f"decision_cb error: {e}")

        # ── Override log callback
        def override_cb(msg):
            try:
                d = json.loads(msg.data)
                with state_lock:
                    state['override_count'] = int(
                        d.get('override_number', 0))
            except Exception as e:
                node.get_logger().warn(f"override_cb error: {e}")

        # ── Health score callback
        def health_cb(msg):
            try:
                if len(msg.data) >= 1:
                    print("RECEIVED HEALTH")
                    score = round(float(msg.data[0]), 1)
                    with state_lock:
                        state['health_history'].append(score)
                        if len(state['health_history']) > 200:
                            state['health_history'].pop(0)
            except Exception as e:
                node.get_logger().warn(f"health_cb error: {e}")

        # ── Create subscriptions with exact topic names
        node.create_subscription(
            String, '/supervisor/safety_status', safety_cb, 10)
        node.create_subscription(
            String, '/supervisor/flow_status', flow_cb, 10)
        node.create_subscription(
            String, '/supervisor/decision', decision_cb, 10)
        node.create_subscription(
            String, '/supervisor/override_log', override_cb, 10)
        node.create_subscription(
            Float32MultiArray, '/localization_health', health_cb, 10)

        ros_ok = True
        node.get_logger().info(
            "Dashboard ROS listener ready — subscribed to 5 topics")

        rclpy.spin(node)

    except Exception as e:
        print(f"[ROS THREAD ERROR] {e}")
        traceback.print_exc()
    finally:
        ros_ok = False
        try:
            rclpy.shutdown()
        except Exception:
            pass


# ── SQLite fallback reader ────────────────────────────────────────────────────

def get_db_decisions(limit=100):
    db_path = os.path.expanduser('~/logs/decisions.db')
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        rows = conn.execute(
            "SELECT timestamp, decision, safety, flow, "
            "overrides, reason "
            "FROM decisions ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        keys = ['timestamp', 'decision', 'safety',
                'flow', 'overrides', 'reason']
        return [dict(zip(keys, row)) for row in rows]
    except Exception as e:
        print(f"DB read error: {e}")
        return []


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/state')
def get_state():
    with state_lock:
        # Return a deep copy to avoid holding lock during JSON serialise
        return jsonify({
            'safety':           dict(state['safety']),
            'flow':             dict(state['flow']),
            'decision':         dict(state['decision']),
            'health_history':   list(state['health_history']),
            'decision_history': list(state['decision_history']),
            'override_count':   state['override_count'],
            'ros_connected':    state['ros_connected'],
            'last_update':      state['last_update'],
        })


@app.route('/api/decisions')
def get_decisions():
    """
    Primary source: live decision_history from ROS.
    Fallback: SQLite database (persists across sessions).
    """
    with state_lock:
        live = list(state['decision_history'])

    if live:
        return jsonify(live[:100])
    else:
        # Fallback to SQLite when ROS not flowing
        return jsonify(get_db_decisions(100))


@app.route('/api/health')
def get_health():
    """Quick endpoint just for health score — used by sparkline."""
    with state_lock:
        return jsonify({
            'history': list(state['health_history']),
            'current': state['health_history'][-1]
                       if state['health_history'] else 0.0,
        })


@app.route('/api/status')
def get_status():
    """Diagnostic endpoint — shows what the backend is receiving."""
    with state_lock:
        return jsonify({
            'ros_connected':       state['ros_connected'],
            'last_update':         state['last_update'],
            'health_samples':      len(state['health_history']),
            'decision_log_length': len(state['decision_history']),
            'current_safety':      state['safety']['status'],
            'current_decision':    state['decision']['decision'],
        })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*55)
    print("AI Supervisory Dashboard")
    print("="*55)
    print("Starting ROS 2 listener thread...")

    t = threading.Thread(target=ros_thread_fn, daemon=True)
    t.start()

    # Give ROS thread 2 seconds to initialise
    time.sleep(2)

    if ros_ok:
        print("ROS listener: CONNECTED")
    else:
        print("ROS listener: starting (topics will connect when nodes run)")

    print("")
    print("Dashboard: http://localhost:5000")
    print("API status: http://localhost:5000/api/status")
    print("="*55)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False   # CRITICAL: reloader breaks ROS thread
    )
