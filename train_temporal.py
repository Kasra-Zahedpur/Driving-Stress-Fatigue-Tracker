# ─── train_temporal.py ────────────────────────────────────────────────────────
# Train an MLP on temporal-window features for driver-distraction detection.
#
# Reads features_temporal.csv (105 features per window) and runs the same
# 5-fold 2-subjects-out CV used for the per-frame models, so results are
# directly comparable.
#
# This is the temporal upgrade to train_features.py. The new "_std" and
# "_range" columns capture MOTION (head turning, hand reaching) which the
# static-frame models had no access to.
# ──────────────────────────────────────────────────────────────────────────────
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

FEATURES_CSV = r"C:\Users\Alex Clare\Downloads\Dataset\features_temporal.csv"
RESULTS_DIR  = r"C:\Users\Alex Clare\Downloads\Dataset\results_temporal"

NUM_EPOCHS    = 60
BATCH_SIZE    = 128
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.05
HIDDEN_DIMS   = [256, 128, 64]
DROPOUT       = 0.4         # higher than per-frame model — dataset is smaller
NUM_CLASSES   = 3
CLASS_NAMES   = ["Baseline", "Distracted Driving", "Safe Driving"]

SUBJECT_PAIRS = [
    ("P5",  "P21"),
    ("P6",  "P26"),
    ("P7",  "P33"),
    ("P19", "P34"),
    ("P20", "P36"),
]

# Columns that are NOT features
META_COLS = {"subject", "session", "label", "first_image", "n_frames"}

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


def fit_epoch(model, loader, opt, crit):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, 100.0 * correct / n


@torch.no_grad()
def eval_epoch(model, loader, crit):
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = crit(out, y)
        total_loss += loss.item() * x.size(0)
        n          += x.size(0)
        preds_all.append(out.argmax(1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    acc = 100.0 * (preds == labels).mean()
    return total_loss / n, acc, preds, labels


def plot_confusion(cm, classes, title, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues"); plt.title(title); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=30, ha="right")
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Loaded {len(df):,} windows from {os.path.basename(FEATURES_CSV)}")
    print(f"Features per window: {len(feature_cols)}")
    print(f"Subjects: {sorted(df['subject'].unique(), key=lambda s: int(s[1:]))}")
    print(f"Class distribution: {df['label'].value_counts().sort_index().to_dict()}\n")

    fold_results = []

    for fold_idx, val_pair in enumerate(SUBJECT_PAIRS, start=1):
        print("=" * 60)
        print(f"FOLD {fold_idx}/5   Val subjects: {val_pair}")
        print("=" * 60)

        val_mask  = df["subject"].isin(val_pair)
        train_df  = df.loc[~val_mask]
        val_df    = df.loc[ val_mask]
        print(f"  Train windows: {len(train_df):>5,}    "
              f"Val windows: {len(val_df):>5,}")

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_val   = val_df  [feature_cols].values
        y_val   = val_df  ["label"].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        train_loader = DataLoader(FeatureDataset(X_train, y_train),
                                  batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        val_loader   = DataLoader(FeatureDataset(X_val,   y_val),
                                  batch_size=BATCH_SIZE, shuffle=False)

        model = MLP(len(feature_cols), HIDDEN_DIMS, NUM_CLASSES,
                    dropout=DROPOUT).to(DEVICE)
        crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS,
                                                     eta_min=LR * 0.01)

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        best = {"val_loss": float("inf"), "epoch": 0,
                "preds": None, "labels": None, "val_acc": 0.0}

        t0 = time.time()
        for epoch in range(1, NUM_EPOCHS + 1):
            tl, ta = fit_epoch(model, train_loader, opt, crit)
            vl, va, p, lb = eval_epoch(model, val_loader, crit)
            sched.step()
            train_losses.append(tl); val_losses.append(vl)
            train_accs.append(ta);   val_accs.append(va)
            tag = ""
            if vl < best["val_loss"]:
                best.update(val_loss=vl, epoch=epoch, preds=p, labels=lb,
                            val_acc=va)
                torch.save(model.state_dict(),
                           os.path.join(RESULTS_DIR, f"mlp_fold{fold_idx}.pth"))
                tag = "  ✅ best"
            print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}  "
                  f"Train {tl:.4f}/{ta:5.2f}%  "
                  f"Val {vl:.4f}/{va:5.2f}%{tag}")

        elapsed = (time.time() - t0) / 60
        print(f"\n  Best Val Loss {best['val_loss']:.4f} @ epoch {best['epoch']}  "
              f"Best Val Acc {max(val_accs):.2f}%   ({elapsed:.1f} min)")

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(train_losses, label="Train"); axes[0].plot(val_losses, label="Val")
        axes[0].set_title(f"Fold {fold_idx} — Loss"); axes[0].legend()
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[1].plot(train_accs,   label="Train"); axes[1].plot(val_accs,   label="Val")
        axes[1].set_title(f"Fold {fold_idx} — Accuracy"); axes[1].legend()
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"curves_fold{fold_idx}.png"))
        plt.close()

        cm = confusion_matrix(best["labels"], best["preds"],
                              labels=list(range(NUM_CLASSES)))
        plot_confusion(cm, CLASS_NAMES,
                       f"Fold {fold_idx} — Best Confusion Matrix",
                       os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}.png"))

        fold_results.append({
            "fold":           fold_idx,
            "val_subjects":   "+".join(val_pair),
            "train_windows":  len(train_df),
            "val_windows":    len(val_df),
            "best_val_loss":  round(best["val_loss"], 4),
            "best_epoch":     best["epoch"],
            "best_val_acc":   round(max(val_accs), 2),
            "epoch1_val_loss":round(val_losses[0], 4),
            "val_loss_drop":  round(val_losses[0] - best["val_loss"], 4),
            "elapsed_min":    round(elapsed, 1),
        })

    print("\n" + "=" * 60)
    print("TEMPORAL CROSS-VALIDATION COMPLETE")
    print("=" * 60)
    summary = pd.DataFrame(fold_results)
    print(summary.to_string(index=False))
    print(f"\nMean Best Val Loss : {summary['best_val_loss'].mean():.4f}")
    print(f"Mean Val Loss Drop : {summary['val_loss_drop'].mean():.4f}")
    print(f"Mean Best Val Acc  : {summary['best_val_acc'].mean():.2f}%   "
          f"(std {summary['best_val_acc'].std():.2f}%)")

    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    try:
        summary.to_excel(os.path.join(RESULTS_DIR, "summary.xlsx"), index=False)
    except ImportError:
        pass


if __name__ == "__main__":
    main()