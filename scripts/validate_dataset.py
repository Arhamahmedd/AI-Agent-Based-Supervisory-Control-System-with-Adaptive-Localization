#!/usr/bin/env python3
"""Day 4 — Dataset validation before training."""

import pandas as pd
import numpy as np
import os

CSV = os.path.expanduser('~/data/combined_dataset.csv')

FEATURES = [
    'acc_x','acc_y','acc_z','gyro_z','linear_x','angular_z',
    'yaw_odom','yaw_filtered','yaw_diff',
    'acc_y_mul_gyro_z','gyro_z_mul_linear_x',
    'cmd_linear_x','cmd_angular_z',
    'delta_angular_z','delta_cmd_angular_z',
]
TARGETS = ['error_x','error_y','error_yaw']

print("Loading dataset...")
df = pd.read_csv(CSV)
print(f"Shape: {df.shape}")

checks = []

# Check 1: All columns present
missing_cols = [c for c in FEATURES + TARGETS if c not in df.columns]
checks.append(('All 18 columns present', len(missing_cols) == 0,
                f"Missing: {missing_cols}" if missing_cols else "OK"))

# Check 2: No NaN values
nan_count = df[FEATURES + TARGETS].isna().sum().sum()
checks.append(('No NaN values', nan_count == 0,
                f"{nan_count} NaN values found" if nan_count else "OK"))

# Check 3: Sufficient samples
checks.append(('Minimum 10000 samples', len(df) >= 10000,
                f"Only {len(df)} rows — re-record bags"))

# Check 4: All profiles present
profiles = df['profile'].unique().tolist()
has_all = all(p in profiles for p in ['static','moderate','aggressive'])
checks.append(('All 3 profiles present', has_all,
                f"Found: {profiles}"))

# Check 5: Error ranges are sensible
max_yaw = df['error_yaw'].abs().max()
checks.append(('error_yaw max < 3.0 rad', max_yaw < 3.0,
                f"Max yaw error: {max_yaw:.4f}"))

# Check 6: Features are not all zero
zero_features = [f for f in FEATURES if df[f].std() < 1e-10]
checks.append(('No zero-variance features', len(zero_features) == 0,
                f"Zero variance: {zero_features}" if zero_features else "OK"))

# Check 7: Targets have variance (not all same value)
zero_targets = [t for t in TARGETS if df[t].std() < 1e-10]
checks.append(('Targets have variance', len(zero_targets) == 0,
                f"Zero variance: {zero_targets}" if zero_targets else "OK"))

print("\nValidation Results:")
print("="*55)
all_pass = True
for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        print(f"         → {detail}")
        all_pass = False

print("="*55)
if all_pass:
    print("ALL CHECKS PASSED — Ready for Day 5 training")
    print(f"\nFinal dataset: {len(df)} rows × {len(FEATURES)} features + {len(TARGETS)} targets")
    print(f"Expected LSTM input shape per sample: [20, {len(FEATURES)}]")
    print(f"Expected LSTM output shape per sample: [{len(TARGETS)}]")
else:
    print("SOME CHECKS FAILED — Fix issues before Day 5")
