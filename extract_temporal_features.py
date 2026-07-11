# ─── extract_temporal_features.py ─────────────────────────────────────────────
# Convert per-frame MediaPipe features into per-WINDOW temporal features.
#
# Why this works: distracted driving is a motion pattern, not a static pose.
# A window of 16 consecutive sampled frames covers ~3-5 seconds of video,
# which is long enough to capture head turns toward a phone, hand reaches,
# repeated glances off-road, etc. We summarise each window with 5 statistics
# per feature (mean/std/min/max/range) — the std and range columns are the
# new signal: they measure MOVEMENT, which is exactly what "distracted" means.
#
# Reads:  features_v1.csv         (per-frame, 21 features)
# Writes: features_temporal.csv   (per-window, 21*5 = 105 features)
#
# Subject split is preserved (each window is fully inside one session of one
# subject), so cross-subject CV still works.
# ──────────────────────────────────────────────────────────────────────────────
import os
import re

import numpy as np
import pandas as pd

INPUT_CSV    = r"C:\Users\Alex Clare\Downloads\Dataset\features_v1.csv"
OUT_CSV      = r"C:\Users\Alex Clare\Downloads\Dataset\features_temporal.csv"

WINDOW_SIZE  = 16   # frames per window (in sampled frames; original = ×20)
STRIDE       = 8    # 50% overlap doubles dataset size without much redundancy


# ──────────────────────────────────────────────────────────────────────────────
# Session identification — uses the image's parent directory + subject as a
# unique session key. Works whether the dataset is laid out as
# Subject/Class/SessionN/frame.jpg or Subject/SessionN_Class/frame.jpg.
# ──────────────────────────────────────────────────────────────────────────────
def session_key(path):
    parent = os.path.dirname(str(path))
    parts  = re.split(r"[\\/]", parent)
    # Use the last two non-empty path components as the session ID
    parts = [p for p in parts if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")


def frame_sort_key(path):
    """Extract any digits from the filename so frames sort numerically."""
    name = os.path.basename(str(path))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows from {os.path.basename(INPUT_CSV)}")

    meta_cols    = ["image_path", "subject", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Per-frame features: {len(feature_cols)}")

    # ── Add session and frame-order columns ──────────────────────────────────
    df["session"]    = df["image_path"].apply(session_key)
    df["_frame_idx"] = df["image_path"].apply(frame_sort_key)

    # Sort within each (subject, session) so frames are in temporal order
    df = df.sort_values(["subject", "session", "_frame_idx"]).reset_index(drop=True)

    # ── Sanity: each session should have a single label ──────────────────────
    label_check = df.groupby(["subject", "session"])["label"].nunique()
    mixed = label_check[label_check > 1]
    if len(mixed) > 0:
        print(f"\n⚠️  {len(mixed)} session(s) have mixed labels — session "
              f"detection may need adjusting. First few:\n{mixed.head()}")
    else:
        print("✓ All sessions have a single class label")

    n_sessions = df.groupby(["subject", "session"]).ngroups
    print(f"\nTotal sessions detected: {n_sessions}")
    print("Per-subject session counts:")
    print(df.groupby("subject")["session"].nunique().to_string())

    # ── Build windows ────────────────────────────────────────────────────────
    print(f"\nBuilding windows (size={WINDOW_SIZE}, stride={STRIDE}) ...")
    windows = []
    for (subject, session), grp in df.groupby(["subject", "session"]):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        if n < WINDOW_SIZE:
            continue
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            w = grp.iloc[start : start + WINDOW_SIZE]
            row = {
                "subject":     subject,
                "session":     session,
                "label":       int(w["label"].iloc[0]),
                "first_image": w["image_path"].iloc[0],
                "n_frames":    WINDOW_SIZE,
            }
            for col in feature_cols:
                v = w[col].values.astype(np.float64)
                row[f"{col}_mean"]  = float(v.mean())
                row[f"{col}_std"]   = float(v.std())
                row[f"{col}_min"]   = float(v.min())
                row[f"{col}_max"]   = float(v.max())
                row[f"{col}_range"] = float(v.max() - v.min())
            windows.append(row)

    out = pd.DataFrame(windows)
    out.to_csv(OUT_CSV, index=False)

    n_feats = len([c for c in out.columns
                   if c not in ("subject", "session", "label",
                                "first_image", "n_frames")])
    print(f"\nSaved {len(out):,} windows → {OUT_CSV}")
    print(f"Features per window: {n_feats}")
    print(f"\nClass distribution:")
    print(out["label"].value_counts().sort_index().to_string())
    print(f"\nWindows per subject:")
    print(out.groupby("subject").size().to_string())


if __name__ == "__main__":
    main()