#!/usr/bin/env python3
"""
Day 5 Part B — LSTM Predictor training.
Architecture: 2-layer LSTM → FC → 3 outputs
Input: [batch, 20, 15] — 20 timestep window, 15 features
Output: [batch, 3] — predicted error_x, error_y, error_yaw
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser('~/data')
MODELS_DIR = os.path.expanduser('~/models')
PLOTS_DIR  = os.path.expanduser('~/plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

WINDOW_SIZE = 20     # 4 seconds at 5Hz — how many timesteps LSTM sees
BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3

# ── GPU setup ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Dataset class ─────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Converts a flat [N, 15] array into sliding windows of shape [N-W, W, 15].
    Each sample is W consecutive timesteps.
    Target is the label at the END of each window.
    
    Why end of window: we want to predict the error at time T
    using data from T-19 through T. The LSTM reads the sequence
    and makes a prediction about the current moment.
    """
    def __init__(self, X, y, window_size=20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.W = window_size

    def __len__(self):
        return len(self.X) - self.W

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.W]          # [W, 15]
        y_target = self.y[idx + self.W]                 # [3]
        return x_window, y_target


# ── Model definition ──────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    Two-layer LSTM followed by two fully-connected layers.
    
    LSTM layer 1 (64 units): Extracts primary temporal patterns.
    LSTM layer 2 (32 units): Refines patterns, adds regularisation via dropout.
    FC layers: Maps temporal features to 3 continuous error predictions.
    
    batch_first=True: input shape is [batch, seq, features] — more intuitive.
    """
    def __init__(self, input_size=15, hidden1=64, hidden2=32,
                 output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        # x shape: [batch, 20, 15]
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch, 20, 64]
        # Take only the last timestep — it encodes the full sequence history
        last_hidden = lstm_out[:, -1, :]
        # last_hidden shape: [batch, 64]
        return self.fc(last_hidden)
        # output shape: [batch, 3]


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading preprocessed data...")
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
y_test_raw   = np.load(os.path.join(DATA_DIR, 'y_test_raw.npy'))
profile_test = np.load(os.path.join(DATA_DIR, 'profile_test.npy'),
                        allow_pickle=True)
scaler_y = joblib.load(os.path.join(MODELS_DIR, 'scaler_y.joblib'))

train_ds = SlidingWindowDataset(X_train, y_train, WINDOW_SIZE)
test_ds  = SlidingWindowDataset(X_test,  y_test,  WINDOW_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"Train windows: {len(train_ds)}")
print(f"Val windows:   {len(test_ds)}")
print(f"Input shape:   [batch, {WINDOW_SIZE}, 15]")


# ── Training setup ────────────────────────────────────────────────────────────

model     = LSTMPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
except TypeError:
    # Older PyTorch versions do not support the verbose argument.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

train_losses = []
val_losses   = []
best_val_loss = float('inf')
best_epoch    = 0

print(f"\nTraining for {EPOCHS} epochs...")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>12}")
print("-" * 45)

for epoch in range(EPOCHS):

    # ── Train phase
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        # This is critical for stable LSTM training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # ── Validation phase
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_batch_losses.append(loss.item())

    val_loss = np.mean(val_batch_losses)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save(model.state_dict(),
                   os.path.join(MODELS_DIR, 'lstm_best.pt'))

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"{epoch:6d} {train_loss:12.6f} {val_loss:12.6f} {current_lr:12.2e}")

print(f"\nBest model: epoch {best_epoch}  val_loss={best_val_loss:.6f}")


# ── Final evaluation ──────────────────────────────────────────────────────────

print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'lstm_best.pt')))
model.eval()

all_preds = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        pred = model(X_batch.to(device)).cpu().numpy()
        all_preds.append(pred)

y_pred_scaled = np.vstack(all_preds)

# Inverse scale to original units
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# y_test_raw is aligned to full test set — trim to match windows
y_test_aligned = y_test_raw[WINDOW_SIZE : WINDOW_SIZE + len(y_pred)]
profiles_aligned = profile_test[WINDOW_SIZE : WINDOW_SIZE + len(y_pred)]

overall_mae = mean_absolute_error(y_test_aligned, y_pred)
overall_r2  = r2_score(y_test_aligned, y_pred)

print(f"\nLSTM Results:")
print(f"  MAE: {overall_mae:.4f}")
print(f"  R²:  {overall_r2:.4f}")

target_names = ['error_x', 'error_y', 'error_yaw']
print(f"\nPer-target:")
for i, name in enumerate(target_names):
    r2  = r2_score(y_test_aligned[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test_aligned[:, i], y_pred[:, i])
    print(f"  {name:12s}: R²={r2:.4f}  MAE={mae:.5f}")

print(f"\nYaw error by motion profile:")
print(f"  {'Profile':12s} {'Median':>10s} {'IQR':>10s} {'Mean':>10s} {'Max':>10s}")
print(f"  {'-'*52}")

for profile in ['static', 'moderate', 'aggressive']:
    mask = profiles_aligned == profile
    if mask.sum() == 0:
        continue
    yaw_error = np.abs(y_pred[mask, 2] - y_test_aligned[mask, 2])
    q1, q3 = np.percentile(yaw_error, [25, 75])
    print(f"  {profile:12s} "
          f"{np.median(yaw_error):10.4f} "
          f"{(q3-q1):10.4f} "
          f"{yaw_error.mean():10.4f} "
          f"{yaw_error.max():10.4f}")


# ── Save training curves plot ─────────────────────────────────────────────────

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses,   label='Val Loss',   linewidth=2)
plt.axvline(best_epoch, color='red', linestyle='--',
            label=f'Best epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (scaled)')
plt.title('LSTM Training Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'lstm_training_curves.png'), dpi=150)
plt.close()
print(f"\nTraining curve saved: ~/plots/lstm_training_curves.png")

# Save final model with metadata
torch.save({
    'epoch': best_epoch,
    'model_state_dict': model.state_dict(),
    'val_loss': best_val_loss,
    'overall_mae': overall_mae,
    'overall_r2': overall_r2,
}, os.path.join(MODELS_DIR, 'lstm_checkpoint.pt'))

print(f"Full checkpoint saved: ~/models/lstm_checkpoint.pt")
print("\nDay 5 complete.")