#!/usr/bin/env python3
"""
Decision Engine — Day 7
Subscribes to all three supervisor topics.
Applies deterministic priority rules every 1 second.
Logs every decision to SQLite for audit trail.
"""

import os
import json
import sqlite3
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

LOGS_DIR = os.path.expanduser('~/logs')


class DecisionEngine(Node):

    EVAL_PERIOD = 1.0   # decision every 1 second

    def __init__(self):
        super().__init__('decision_engine')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )])

        os.makedirs(LOGS_DIR, exist_ok=True)

        # ── State from agents
        self.safety_status = 'UNKNOWN'
        self.flow_status   = 'UNKNOWN'
        self.override_count = 0
        self.safety_reason  = ''
        self.flow_reason    = ''
        self.safety_health  = None

        # ── Subscriptions
        self.create_subscription(
            String, '/supervisor/safety_status',
            self.safety_cb, 10)
        self.create_subscription(
            String, '/supervisor/flow_status',
            self.flow_cb, 10)
        self.create_subscription(
            String, '/supervisor/override_log',
            self.override_cb, 10)

        # ── Publisher
        self.decision_pub = self.create_publisher(
            String, '/supervisor/decision', 10)

        # ── Decision timer
        self.create_timer(self.EVAL_PERIOD, self.make_decision)

        # ── SQLite audit log
        db_path = os.path.join(LOGS_DIR, 'decisions.db')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                decision  TEXT,
                safety    TEXT,
                flow      TEXT,
                overrides INTEGER,
                reason    TEXT
            )
        ''')
        self.conn.commit()
        self.get_logger().info(f"Decision Engine ready. DB: {db_path}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def safety_cb(self, msg):
        try:
            d = json.loads(msg.data)
            self.safety_status = d.get('status', 'UNKNOWN')
            self.safety_reason = d.get('reason', '')
            health = d.get('health_score', None)
            self.safety_health = float(health) if health is not None else None
        except json.JSONDecodeError:
            pass

    def flow_cb(self, msg):
        try:
            d = json.loads(msg.data)
            self.flow_status = d.get('status', 'UNKNOWN')
            self.flow_reason = d.get('recommendation', '')
        except json.JSONDecodeError:
            pass

    def override_cb(self, msg):
        try:
            d = json.loads(msg.data)
            self.override_count = d.get('override_number', self.override_count)
        except json.JSONDecodeError:
            pass

    # ── Decision logic ────────────────────────────────────────────────────────

    def make_decision(self):
        if self.safety_status == 'UNKNOWN':
            return

        # Deterministic priority rules — AI advises, rules decide
        if self.safety_status == 'RED':
            # Demo-tuned rule: hard stop only for very low health.
            if self.safety_health is not None and self.safety_health < 2.0:
                decision = 'STOP'
                reason   = f"Critical localization failure. {self.safety_reason}"
            else:
                decision = 'REDUCE_SPEED'
                reason   = (f"Safety is RED but health is above hard-stop floor "
                            f"(health={self.safety_health}). Proceeding cautiously.")

        elif self.safety_status == 'AMBER' and self.flow_status == 'DEGRADED':
            decision = 'REDUCE_SPEED'
            reason   = (f"Elevated uncertainty + degraded flow. "
                        f"{self.flow_reason}")

        elif self.safety_status == 'AMBER':
            decision = 'REDUCE_SPEED'
            reason   = f"Elevated localization uncertainty. {self.safety_reason}"

        elif self.override_count > 5:
            decision = 'FLAG_MAINTENANCE'
            reason   = (f"High manual override count ({self.override_count}). "
                        "System review recommended.")

        else:
            decision = 'CONTINUE'
            reason   = (f"All systems nominal. "
                        f"Safety={self.safety_status} "
                        f"Flow={self.flow_status}")

        ts = datetime.now().isoformat()

        # ── Publish decision
        payload = {
            'decision':  decision,
            'reason':    reason,
            'safety':    self.safety_status,
            'flow':      self.flow_status,
            'overrides': self.override_count,
            'timestamp': ts,
        }
        out = String()
        out.data = json.dumps(payload)
        self.decision_pub.publish(out)

        # ── Log to SQLite
        self.conn.execute(
            "INSERT INTO decisions VALUES (NULL,?,?,?,?,?,?)",
            (ts, decision, self.safety_status,
             self.flow_status, self.override_count, reason)
        )
        self.conn.commit()

        # ── Console log
        if decision == 'CONTINUE':
            self.get_logger().info(
                f"[DECISION] {decision}",
                throttle_duration_sec=5.0)
        elif decision == 'STOP':
            self.get_logger().error(f"[DECISION] {decision} — {reason}")
        else:
            self.get_logger().warn(f"[DECISION] {decision} — {reason}")

    def destroy_node(self):
        if hasattr(self, 'conn'):
            self.conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DecisionEngine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
