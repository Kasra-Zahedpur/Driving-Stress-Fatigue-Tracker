# ─── prepare_deployment.py ────────────────────────────────────────────────────
# Train the final temporal MLP on ALL windowed features (no holdout) and save
# a single deployable bundle containing the model weights, the standardiser
# stats, the feature column order, and class labels.
#
# Why "no holdout": the cross-validation in train_temporal.py already proved
# the architecture works (63% mean val acc across 5 subject-disjoint folds).
# For the deployed model, we want it trained on every available window so the
# demo benefits from the full ~1700 examples instead of 80% of them.
#
# Output: deployment/model_bundle.pt — single file the demo loads.
# ──────────────────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = r"C:\Users\Alex Clare\Downloads\Dataset\features_temporal.csv"
OUTPUT_DIR   = r"C:\Users\Alex Clare\Downloads\Dataset\deployment"
BUNDLE_PATH  = os.path.join(OUTPUT_DIR, "model_bundle.pt")

NUM_EPOCHS    = 60
BATCH_SIZE    = 128
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.12       # bumped from 0.05 — softens probability collapse
HIDDEN_DIMS   = [256, 128, 64]
DROPOUT       = 0.4
NUM_CLASSES   = 3
CLASS_NAMES   = ["Baseline", "Distracted Driving", "Safe Driving"]
META_COLS     = {"subject", "session", "label", "first_image", "n_frames"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_classes, dropout=0.4):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Training final model on ALL {len(df):,} windows  "
          f"({len(feature_cols)} features)")
    print(f"Class distribution: {df['label'].value_counts().sort_index().to_dict()}")

    X = df[feature_cols].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loader = DataLoader(
        FeatureDataset(X_scaled, y),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    model = MLP(len(feature_cols), HIDDEN_DIMS, NUM_CLASSES,
                dropout=DROPOUT).to(DEVICE)
    crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS,
                                                 eta_min=LR * 0.01)

    print(f"\nDevice: {DEVICE}")
    print("Training:")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, yy in loader:
            x, yy = x.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, yy)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == yy).sum().item()
            n          += x.size(0)
        sched.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}  "
                  f"Loss {total_loss/n:.4f}  Acc {100*correct/n:.2f}%")

    # ── Save bundle ──────────────────────────────────────────────────────────
    torch.save({
        "state_dict":   model.state_dict(),
        "scaler_mean":  scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "feature_cols": feature_cols,
        "class_names":  CLASS_NAMES,
        "hidden_dims":  HIDDEN_DIMS,
        "dropout":      DROPOUT,
        "num_classes":  NUM_CLASSES,
        "window_size":  16,
    }, BUNDLE_PATH)

    size_mb = os.path.getsize(BUNDLE_PATH) / 1e6
    print(f"\nSaved deployment bundle → {BUNDLE_PATH}  ({size_mb:.2f} MB)")
    print("\nReady for demo. Run: python demo.py")


if __name__ == "__main__":
    main()