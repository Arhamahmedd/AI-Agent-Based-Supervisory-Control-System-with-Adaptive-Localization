#!/usr/bin/env python3
"""
Day 5 Part A — Random Forest Regressor baseline.
This replicates the model from Abaza (2025).
Target: MAE ~0.0061, R² ~0.989 on combined test set.
If you achieve this, your dataset pipeline matches Paper 1.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # no display needed — saves to file
import matplotlib.pyplot as plt

DATA_DIR   = os.path.expanduser('~/data')
MODELS_DIR = os.path.expanduser('~/models')
PLOTS_DIR  = os.path.expanduser('~/plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

scaler_y = joblib.load(os.path.join(MODELS_DIR, 'scaler_y.joblib'))

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
y_test_raw    = np.load(os.path.join(DATA_DIR, 'y_test_raw.npy'))
profile_test  = np.load(os.path.join(DATA_DIR, 'profile_test.npy'), allow_pickle=True)

print("Training Random Forest Regressor...")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples:  {len(X_test)}")

# ── Train ─────────────────────────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,          # no depth limit — matches Paper 1 defaults
    min_samples_split=2,
    max_features='sqrt',     # Paper 1 uses scikit-learn defaults
    random_state=42,
    n_jobs=-1,               # use all CPU cores
    oob_score=True,          # out-of-bag validation during training
    verbose=1,
)
rf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_scaled = rf.predict(X_test)

# Convert back to original units for meaningful metrics
y_pred = scaler_y.inverse_transform(y_pred_scaled)

overall_mae = mean_absolute_error(y_test_raw, y_pred)
overall_r2  = r2_score(y_test_raw, y_pred)

print(f"\nOverall Results:")
print(f"  MAE: {overall_mae:.4f}  (Paper 1 target: ~0.0061)")
print(f"  R²:  {overall_r2:.4f}  (Paper 1 target: ~0.989)")
print(f"  OOB score: {rf.oob_score_:.4f}")

target_names = ['error_x', 'error_y', 'error_yaw']
print(f"\nPer-target R²:")
for i, name in enumerate(target_names):
    r2 = r2_score(y_test_raw[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test_raw[:, i], y_pred[:, i])
    print(f"  {name:12s}: R²={r2:.4f}  MAE={mae:.5f}")

# ── Per-profile evaluation ────────────────────────────────────────────────────
print(f"\nYaw error by motion profile:")
print(f"  {'Profile':12s} {'Median':>10s} {'IQR':>10s} {'Mean':>10s} {'Max':>10s}")
print(f"  {'-'*52}")

for profile in ['static', 'moderate', 'aggressive']:
    mask = profile_test == profile
    if mask.sum() == 0:
        continue
    yaw_pred = y_pred[mask, 2]
    yaw_true = y_test_raw[mask, 2]
    yaw_error = np.abs(yaw_pred - yaw_true)
    q1, q3 = np.percentile(yaw_error, [25, 75])
    print(f"  {profile:12s} "
          f"{np.median(yaw_error):10.4f} "
          f"{(q3-q1):10.4f} "
          f"{yaw_error.mean():10.4f} "
          f"{yaw_error.max():10.4f}")

# Paper 1 reference values for comparison
print(f"\n  Paper 1 baseline (for comparison):")
print(f"  {'static':12s}  median=0.0362  IQR=0.0489")
print(f"  {'moderate':12s}  median=0.0381  IQR=0.1069")
print(f"  {'aggressive':12s}  mean=0.2257   max=0.9491")

# ── Feature importance ────────────────────────────────────────────────────────
FEATURES = [
    'acc_x','acc_y','acc_z','gyro_z','linear_x','angular_z',
    'yaw_odom','yaw_filtered','yaw_diff',
    'acc_y_mul_gyro_z','gyro_z_mul_linear_x',
    'cmd_linear_x','cmd_angular_z',
    'delta_angular_z','delta_cmd_angular_z',
]

importances = rf.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

print(f"\nTop 5 most important features:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"  {i+1}. {FEATURES[idx]:30s}: {importances[idx]:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, 'rf_baseline.joblib')
joblib.dump(rf, model_path)
print(f"\nModel saved: {model_path}")

# ── Plot: feature importances ─────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.barh(
    [FEATURES[i] for i in sorted_idx],
    [importances[i] for i in sorted_idx],
    color='steelblue'
)
plt.xlabel('Feature Importance')
plt.title('Random Forest — Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rf_feature_importance.png'), dpi=150)
plt.close()
print(f"Plot saved: ~/plots/rf_feature_importance.png")

print("\nRandom Forest baseline training complete.")
