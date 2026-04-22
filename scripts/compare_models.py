#!/usr/bin/env python3
"""Day 5 — Final comparison: LSTM vs Random Forest vs Paper 1 baseline."""

import os
import numpy as np
import joblib
import torch
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = os.path.expanduser('~/data')
MODELS_DIR = os.path.expanduser('~/models')
PLOTS_DIR  = os.path.expanduser('~/plots')

WINDOW_SIZE = 20
FEATURES_N  = 15

# Load data
X_test       = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test       = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
y_test_raw   = np.load(os.path.join(DATA_DIR, 'y_test_raw.npy'))
profile_test = np.load(os.path.join(DATA_DIR, 'profile_test.npy'),
                        allow_pickle=True)
scaler_y = joblib.load(os.path.join(MODELS_DIR, 'scaler_y.joblib'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── RF predictions ────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
rf = joblib.load(os.path.join(MODELS_DIR, 'rf_baseline.joblib'))
rf_pred_scaled = rf.predict(X_test)
rf_pred = scaler_y.inverse_transform(rf_pred_scaled)

# ── LSTM predictions ──────────────────────────────────────────────────────────
class LSTMPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(15, 64, 2, dropout=0.2, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 3))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMPredictor().to(device)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'lstm_best.pt')))
model.eval()

# Build windows for LSTM
X_tensor = torch.FloatTensor(X_test)
lstm_preds = []
with torch.no_grad():
    for i in range(len(X_test) - WINDOW_SIZE):
        window = X_tensor[i:i+WINDOW_SIZE].unsqueeze(0).to(device)
        pred = model(window).cpu().numpy()
        lstm_preds.append(pred[0])

lstm_pred_scaled = np.array(lstm_preds)
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)

# Align all arrays to window-adjusted length
n = len(lstm_pred)
rf_pred_aligned  = rf_pred[WINDOW_SIZE:WINDOW_SIZE+n]
y_true_aligned   = y_test_raw[WINDOW_SIZE:WINDOW_SIZE+n]
profiles_aligned = profile_test[WINDOW_SIZE:WINDOW_SIZE+n]

# ── Print comparison table ────────────────────────────────────────────────────
profiles = ['static', 'moderate', 'aggressive']

print("\n" + "="*75)
print("COMPARISON: Random Forest vs LSTM vs Paper 1 Baseline")
print("="*75)
print(f"\n{'Metric':<20} {'Paper 1 RF':>12} {'Your RF':>12} {'Your LSTM':>12}")
print("-"*58)

# Overall MAE
rf_mae   = mean_absolute_error(y_true_aligned, rf_pred_aligned)
lstm_mae = mean_absolute_error(y_true_aligned, lstm_pred)
print(f"{'Overall MAE':<20} {'0.0061':>12} {rf_mae:>12.4f} {lstm_mae:>12.4f}")

# Overall R²
rf_r2   = r2_score(y_true_aligned, rf_pred_aligned)
lstm_r2 = r2_score(y_true_aligned, lstm_pred)
print(f"{'Overall R²':<20} {'0.989':>12} {rf_r2:>12.4f} {lstm_r2:>12.4f}")

# Yaw error per profile
paper1_ref = {
    'static':     (0.0362, 0.0489),
    'moderate':   (0.0381, 0.1069),
    'aggressive': (None,   None),
}

print(f"\nYaw Error — Median (rad):")
print(f"  {'Profile':<12} {'Paper1 RF':>12} {'Your RF':>12} {'Your LSTM':>12}")
print(f"  {'-'*50}")

for profile in profiles:
    mask = profiles_aligned == profile
    if mask.sum() == 0:
        continue
    rf_yaw_err   = np.abs(rf_pred_aligned[mask, 2] - y_true_aligned[mask, 2])
    lstm_yaw_err = np.abs(lstm_pred[mask, 2] - y_true_aligned[mask, 2])
    ref_med = f"{paper1_ref[profile][0]:.4f}" if paper1_ref[profile][0] else "N/A"
    print(f"  {profile:<12} {ref_med:>12} "
          f"{np.median(rf_yaw_err):>12.4f} "
          f"{np.median(lstm_yaw_err):>12.4f}")

print("\nYaw Error — IQR (rad):")
print(f"  {'Profile':<12} {'Paper1 RF':>12} {'Your RF':>12} {'Your LSTM':>12}")
print(f"  {'-'*50}")

for profile in profiles:
    mask = profiles_aligned == profile
    if mask.sum() == 0:
        continue
    rf_yaw_err   = np.abs(rf_pred_aligned[mask, 2] - y_true_aligned[mask, 2])
    lstm_yaw_err = np.abs(lstm_pred[mask, 2] - y_true_aligned[mask, 2])
    rf_iqr   = np.percentile(rf_yaw_err, 75) - np.percentile(rf_yaw_err, 25)
    lstm_iqr = np.percentile(lstm_yaw_err, 75) - np.percentile(lstm_yaw_err, 25)
    ref_iqr = f"{paper1_ref[profile][1]:.4f}" if paper1_ref[profile][1] else "N/A"
    print(f"  {profile:<12} {ref_iqr:>12} "
          f"{rf_iqr:>12.4f} "
          f"{lstm_iqr:>12.4f}")

# ── Box plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Yaw Prediction Error by Profile\n(RF vs LSTM)', fontsize=14)

for ax, profile in zip(axes, profiles):
    mask = profiles_aligned == profile
    if mask.sum() == 0:
        ax.set_title(f'{profile}\n(no data)')
        continue
    rf_err   = np.abs(rf_pred_aligned[mask, 2] - y_true_aligned[mask, 2])
    lstm_err = np.abs(lstm_pred[mask, 2] - y_true_aligned[mask, 2])
    ax.boxplot([rf_err, lstm_err],
               labels=['Random\nForest', 'LSTM'],
               notch=False, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{profile.capitalize()}')
    ax.set_ylabel('|Yaw Error| (rad)')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rf_vs_lstm_boxplot.png'), dpi=150)
plt.close()
print(f"\nBox plot saved: ~/plots/rf_vs_lstm_boxplot.png")
print("\nComparison complete.")
