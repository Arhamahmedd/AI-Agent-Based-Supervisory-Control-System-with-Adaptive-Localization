#!/usr/bin/env python3
"""
Day 5 — Shared data preparation.
Creates scaled tensors and train/test split used by BOTH models.
Run this ONCE — both models load from the saved files.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser('~/data')
MODELS_DIR = os.path.expanduser('~/models')
os.makedirs(MODELS_DIR, exist_ok=True)

CSV = os.path.join(DATA_DIR, 'combined_dataset.csv')

FEATURES = [
    'acc_x','acc_y','acc_z','gyro_z','linear_x','angular_z',
    'yaw_odom','yaw_filtered','yaw_diff',
    'acc_y_mul_gyro_z','gyro_z_mul_linear_x',
    'cmd_linear_x','cmd_angular_z',
    'delta_angular_z','delta_cmd_angular_z',
]
TARGETS = ['error_x','error_y','error_yaw']

print("Loading dataset...")
df = pd.read_csv(CSV).dropna()
print(f"Total samples: {len(df)}")

X = df[FEATURES].values   # shape: [N, 15]
y = df[TARGETS].values     # shape: [N, 3]

# ── Scale features and targets ────────────────────────────────────────────────
# Why scale: LSTM and gradient-based optimisers converge much faster
# when all inputs are on the same scale (mean≈0, std≈1).
# Random Forest does NOT need scaling but we apply it anyway so both
# models train on identical preprocessed data.
# CRITICAL: fit scalers on TRAINING data only, transform both train+test.

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_test  = scaler_X.transform(X_test_raw)       # transform only — no fit

y_train = scaler_y.fit_transform(y_train_raw)
y_test  = scaler_y.transform(y_test_raw)

# Save scalers — the ROS 2 LSTM node needs these at inference time
joblib.dump(scaler_X, os.path.join(MODELS_DIR, 'scaler_X.joblib'))
joblib.dump(scaler_y, os.path.join(MODELS_DIR, 'scaler_y.joblib'))
print(f"Scalers saved to {MODELS_DIR}")

# Save raw (unscaled) test arrays for evaluation
# We evaluate in original units (metres, radians) — not scaled units
np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(DATA_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(DATA_DIR, 'y_test.npy'),  y_test)
np.save(os.path.join(DATA_DIR, 'y_test_raw.npy'), y_test_raw)  # for metrics

# Also save profile labels for per-regime evaluation
profile_labels = df['profile'].values
_, profile_test = train_test_split(
    profile_labels, test_size=0.1, random_state=42, shuffle=True
)
np.save(os.path.join(DATA_DIR, 'profile_test.npy'), profile_test)

print(f"\nShapes:")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"\nScaler statistics (features):")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:30s} mean={scaler_X.mean_[i]:+.5f}  "
          f"std={scaler_X.scale_[i]:.5f}")

print("\nData preparation complete.")
