# ─── demo.py ──────────────────────────────────────────────────────────────────
# Real-time driver-distraction detector using webcam (or a recorded video file
# if VIDEO_FILE is set below).
#
# Pipeline at runtime:
#   1. Capture frame from webcam/video
#   2. MediaPipe Tasks → 21 per-frame features (head pose, gaze, mouth, hands,
#      shoulder angle)
#   3. Push into rolling 16-frame buffer
#   4. When buffer is full, compute 5 stats per feature → 105 temporal features
#   5. Standardise with saved scaler, run MLP → softmax over 3 classes
#   6. Apply temperature scaling + EMA smoothing + hysteresis on predictions
#   7. Track distraction duration → severity grading (Glance / Distracted /
#      Severe), trigger audio alerts only on sustained distraction
#   8. Overlay banner + bars + duration + FPS badge
#
# Press Q (or ESC) to quit. Press M to mute/unmute audio.
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# Audio (Windows). On non-Windows systems, audio alerts are silently disabled.
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Reuse helpers from extract_features.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_features import (
    ensure_models, make_landmarkers,
    LM, LEFT_IRIS, RIGHT_IRIS, EMPTY_FEATS,
    head_pose_pyr, eye_aspect_ratio, iris_offset, mouth_open_ratio, lm_xy,
)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
BUNDLE_PATH  = r"C:\Users\Alex Clare\Downloads\Dataset\deployment\model_bundle.pt"
WINDOW_SIZE  = 16
VIDEO_FILE   = None     # set to a path to demo on a recording instead of webcam
WEBCAM_INDEX = 0

# ── Probability calibration ─────────────────────────────────────────────────
TEMPERATURE  = 1.5      # divides logits before softmax — softer probabilities
EMA_ALPHA    = 0.30     # weight of the new prediction in the running average

# ── Hysteresis (prevents class flip-flop) ───────────────────────────────────
MIN_SWITCH_FRAMES = 5   # consecutive frames of new class needed to switch

# ── Severity thresholds (seconds of sustained Distracted Driving) ───────────
GLANCE_START_S      = 1.0    # 0–1 s: silent monitoring (could be a mirror check)
DISTRACTED_START_S  = 3.0    # 1–3 s: GLANCE  (visual only, no audio)
SEVERE_START_S      = 5.0    # 3–5 s: DISTRACTED (single beep, red banner)
                             # 5+ s: SEVERE (repeating beep, bright banner)

SEVERE_BEEP_INTERVAL_S = 2.0

# ── Audio frequencies (Hz, ms) ──────────────────────────────────────────────
DISTRACTED_BEEP = (1000, 220)
SEVERE_BEEP     = (1300, 380)

# ── Distracted class detection (which CLASS_NAMES entry is "Distracted") ────
DISTRACTED_LABEL = "Distracted Driving"

# ── Calibration ─────────────────────────────────────────────────────────────
# 5 seconds of "look forward, hands on wheel" at startup. We average the
# user's neutral state and subtract it from the pose-orientation features
# only (head angles, gaze offsets, shoulder angle). Features that are
# physiological / absolute (eye openness, mouth open, hand-to-face distance)
# are NOT calibrated — those should mean the same thing across users.
CALIBRATION_DURATION_S = 5.0
CALIBRATION_FEATURES = [
    "pitch", "yaw", "roll",
    "iris_lx", "iris_ly", "iris_rx", "iris_ry",
    "shoulder_angle",
]
SKIP_CALIBRATION_FOR_VIDEO = True   # video files don't need calibration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture (must match prepare_deployment.py)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Audio helper — non-blocking beep
# ══════════════════════════════════════════════════════════════════════════════
def beep_async(frequency_hz, duration_ms):
    if not AUDIO_AVAILABLE:
        return
    def _beep():
        try:
            winsound.Beep(int(frequency_hz), int(duration_ms))
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame feature extraction (numpy frame in, dict out)
# ══════════════════════════════════════════════════════════════════════════════
def extract_from_frame(frame_bgr, face_lm, pose_lm, hand_lm):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_res = face_lm.detect(mp_img)
    pose_res = pose_lm.detect(mp_img)
    hand_res = hand_lm.detect(mp_img)

    f = dict(EMPTY_FEATS)
    nose_norm = None

    if face_res.face_landmarks:
        flms = face_res.face_landmarks[0]
        f["face_detected"] = 1
        try:
            f["pitch"], f["yaw"], f["roll"] = head_pose_pyr(flms, w, h)
        except Exception:
            pass
        f["left_ear"]  = eye_aspect_ratio(
            flms, LM["left_eye_top"], LM["left_eye_bottom"],
            LM["left_eye_outer"], LM["left_eye_inner"], w, h)
        f["right_ear"] = eye_aspect_ratio(
            flms, LM["right_eye_top"], LM["right_eye_bottom"],
            LM["right_eye_outer"], LM["right_eye_inner"], w, h)
        if len(flms) >= 478:
            f["iris_lx"], f["iris_ly"] = iris_offset(
                flms, LEFT_IRIS,
                LM["left_eye_outer"], LM["left_eye_inner"], w, h)
            f["iris_rx"], f["iris_ry"] = iris_offset(
                flms, RIGHT_IRIS,
                LM["right_eye_outer"], LM["right_eye_inner"], w, h)
        f["mouth_open"] = mouth_open_ratio(flms, w, h)
        nose_xy = lm_xy(flms, LM["nose"], w, h)
        nose_norm = nose_xy / np.array([w, h])

    if hand_res.hand_landmarks:
        for idx, lms in enumerate(hand_res.hand_landmarks):
            try:
                cls_name = hand_res.handedness[idx][0].category_name.lower()
            except (AttributeError, IndexError):
                cls_name = "left" if idx == 0 else "right"
            side = "left" if cls_name == "left" else "right"
            wrist = lms[0]
            f[f"{side}_hand_detected"] = 1
            f[f"{side}_hand_x"] = float(wrist.x)
            f[f"{side}_hand_y"] = float(wrist.y)
            if nose_norm is not None:
                f[f"{side}_hand_to_face"] = float(np.linalg.norm(
                    np.array([wrist.x, wrist.y]) - nose_norm
                ))

    if pose_res.pose_landmarks:
        plms = pose_res.pose_landmarks[0]
        f["pose_detected"] = 1
        ls = plms[11]
        rs = plms[12]
        f["shoulder_angle"] = float(np.degrees(
            np.arctan2(ls.y - rs.y, ls.x - rs.x)
        ))

    return f


# ══════════════════════════════════════════════════════════════════════════════
# Window stats: 5 numbers per raw feature, in the same order training used
# ══════════════════════════════════════════════════════════════════════════════
def compute_window_stats(buffer, raw_feature_names):
    arr = np.array(
        [[f[name] for name in raw_feature_names] for f in buffer],
        dtype=np.float64,
    )
    means  = arr.mean(axis=0)
    stds   = arr.std(axis=0)
    mins   = arr.min(axis=0)
    maxs   = arr.max(axis=0)
    ranges = maxs - mins
    return np.column_stack([means, stds, mins, maxs, ranges]).flatten()


def parse_raw_feature_order(temporal_cols):
    raw, seen = [], set()
    for col in temporal_cols:
        for suffix in ("_mean", "_std", "_min", "_max", "_range"):
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                if base not in seen:
                    seen.add(base)
                    raw.append(base)
                break
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration(cap, face_lm, pose_lm, hand_lm,
                    duration_s=CALIBRATION_DURATION_S, mirror=True):
    """
    Capture per-frame features for `duration_s` seconds while showing a
    fullscreen calibration overlay. Returns a dict mapping feature name →
    baseline mean for the features in CALIBRATION_FEATURES. Returns None
    if the user pressed Q/ESC during calibration.
    """
    print(f"\n── Calibration ({duration_s:.0f}s) ──")
    print("   Look forward, hands on the wheel, eyes on the road.\n")

    samples   = []
    t_start   = time.time()
    t_end     = t_start + duration_s

    while time.time() < t_end:
        ret, frame = cap.read()
        if not ret:
            return None
        if mirror:
            frame = cv2.flip(frame, 1)

        feats = extract_from_frame(frame, face_lm, pose_lm, hand_lm)
        if feats["face_detected"] == 1:
            samples.append(feats)

        # ── Overlay ──────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        # darken background a bit so text is readable
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        elapsed   = time.time() - t_start
        remaining = max(duration_s - elapsed, 0.0)
        progress  = min(elapsed / duration_s, 1.0)

        # Big title
        title = "CALIBRATING"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        cv2.putText(frame, title, ((w - tw) // 2, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3,
                    cv2.LINE_AA)
        # Instruction
        instr = "Look forward — hands on wheel — eyes on road"
        (tw, _), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, instr, ((w - tw) // 2, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2,
                    cv2.LINE_AA)
        # Countdown
        countdown = f"{remaining:.1f}s"
        (tw, _), _ = cv2.getTextSize(countdown, cv2.FONT_HERSHEY_SIMPLEX,
                                     1.2, 3)
        cv2.putText(frame, countdown, ((w - tw) // 2, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                    cv2.LINE_AA)
        # Progress bar
        bar_x1, bar_x2 = 60, w - 60
        bar_y1 = h // 2 + 80
        bar_y2 = bar_y1 + 22
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2),
                      (40, 40, 40), -1)
        fill_w = int((bar_x2 - bar_x1) * progress)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2),
                      (60, 200, 80), -1)
        # Sample counter
        sample_text = f"{len(samples)} valid samples"
        cv2.putText(frame, sample_text, (bar_x1, bar_y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)

        cv2.imshow("Driver Distraction Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return None

    if len(samples) < 10:
        print(f"   ⚠ Only {len(samples)} valid samples — face not detected "
              f"reliably. Skipping calibration.")
        return None

    baseline = {}
    for name in CALIBRATION_FEATURES:
        vals = np.array([s[name] for s in samples], dtype=np.float64)
        baseline[name] = float(vals.mean())

    print(f"   ✓ Calibration complete ({len(samples)} samples).")
    for name, val in baseline.items():
        print(f"     {name:>16s}: {val:+8.3f}")
    return baseline


def apply_calibration(feats, baseline):
    """Subtract baseline mean from selected pose features (in place safe)."""
    if baseline is None:
        return feats
    f = dict(feats)
    for name, base in baseline.items():
        if name in f:
            f[name] = float(f[name]) - base
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Severity model
# ══════════════════════════════════════════════════════════════════════════════
def classify_severity(distracted_duration_s):
    """Map seconds-in-distracted-state to a severity tier."""
    if distracted_duration_s < GLANCE_START_S:
        return "MONITORING"
    if distracted_duration_s < DISTRACTED_START_S:
        return "GLANCE"
    if distracted_duration_s < SEVERE_START_S:
        return "DISTRACTED"
    return "SEVERE"


# ══════════════════════════════════════════════════════════════════════════════
# UI overlay
# ══════════════════════════════════════════════════════════════════════════════
CLASS_COLORS_BGR = {
    "Baseline":            (180, 130,  90),    # blue-grey
    "Distracted Driving":  ( 60,  60, 220),    # red (default — overridden by severity)
    "Safe Driving":        ( 80, 180,  80),    # green
}

SEVERITY_COLORS_BGR = {
    "MONITORING": ( 60,  60, 220),    # red (same as default Distracted)
    "GLANCE":     ( 50, 165, 245),    # orange
    "DISTRACTED": ( 40,  40, 220),    # red
    "SEVERE":     ( 30,  30, 255),    # bright red
}

SEVERITY_LABELS = {
    "MONITORING": "",
    "GLANCE":     "GLANCE",
    "DISTRACTED": "DISTRACTED",
    "SEVERE":     "SEVERE",
}


def draw_overlay(frame, class_names, displayed_class, ema_probs,
                 severity, distracted_duration_s,
                 buffer_len, fps, window_size, audio_muted):
    h, w = frame.shape[:2]

    # ── Top banner ────────────────────────────────────────────────────────
    if buffer_len < window_size:
        banner_color = (90, 90, 90)
        banner_text  = f"Calibrating... ({buffer_len}/{window_size})"
    elif displayed_class is not None:
        cls_name = class_names[displayed_class]
        is_distracted = (cls_name == DISTRACTED_LABEL)
        if is_distracted and severity in SEVERITY_COLORS_BGR:
            banner_color = SEVERITY_COLORS_BGR[severity]
            sev_label = SEVERITY_LABELS.get(severity, "")
            if sev_label:
                banner_text = (f"{cls_name}  -  {sev_label}   "
                               f"({distracted_duration_s:.1f}s)")
            else:
                banner_text = (f"{cls_name}   "
                               f"{ema_probs[displayed_class]*100:.0f}%")
        else:
            banner_color = CLASS_COLORS_BGR.get(cls_name, (100, 100, 100))
            banner_text  = (f"{cls_name}   "
                            f"{ema_probs[displayed_class]*100:.0f}%")
    else:
        banner_color = (90, 90, 90)
        banner_text  = "..."

    banner_h = 60
    cv2.rectangle(frame, (0, 0), (w, banner_h), banner_color, -1)
    cv2.putText(frame, banner_text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                cv2.LINE_AA)

    # ── Probability bars (bottom-left) ────────────────────────────────────
    if ema_probs is not None:
        bar_x, bar_w_full, bar_h = 20, 280, 22
        bar_y = h - 30 - len(class_names) * 32
        for i, name in enumerate(class_names):
            p     = float(ema_probs[i])
            color = CLASS_COLORS_BGR.get(name, (130, 130, 130))
            top   = bar_y + i * 32
            cv2.rectangle(frame, (bar_x, top),
                          (bar_x + bar_w_full, top + bar_h),
                          (40, 40, 40), -1)
            cv2.rectangle(frame, (bar_x, top),
                          (bar_x + int(p * bar_w_full), top + bar_h),
                          color, -1)
            cv2.putText(frame, f"{name}: {p*100:5.1f}%",
                        (bar_x + 8, top + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # ── FPS as small badge in bottom-right ────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 8
    bx2, by2 = w - 12, h - 12
    bx1, by1 = bx2 - tw - 2 * pad, by2 - th - 2 * pad
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
    cv2.putText(frame, fps_text, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1,
                cv2.LINE_AA)

    # ── Audio mute indicator (top-right corner) ───────────────────────────
    if audio_muted:
        mute_text = "MUTED"
        (tw, th), _ = cv2.getTextSize(mute_text, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, 1)
        cv2.rectangle(frame, (w - tw - 24, 70),
                      (w - 8, 70 + th + 12), (40, 40, 40), -1)
        cv2.putText(frame, mute_text, (w - tw - 16, 70 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 200, 80), 1,
                    cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(BUNDLE_PATH):
        print(f"ERROR: model bundle not found at {BUNDLE_PATH}")
        print("Run prepare_deployment.py first.")
        return

    print(f"Loading deployment bundle → {BUNDLE_PATH}")
    bundle = torch.load(BUNDLE_PATH, map_location=DEVICE, weights_only=False)
    feature_cols = bundle["feature_cols"]
    class_names  = bundle["class_names"]
    window_size  = bundle.get("window_size", WINDOW_SIZE)
    num_classes  = bundle["num_classes"]

    raw_feature_names = parse_raw_feature_order(feature_cols)
    print(f"  Raw features ({len(raw_feature_names)})")
    print(f"  Temporal features: {len(feature_cols)}")
    print(f"  Window size: {window_size}")
    print(f"  Classes: {class_names}")

    if DISTRACTED_LABEL in class_names:
        distracted_idx = class_names.index(DISTRACTED_LABEL)
    else:
        distracted_idx = 1
        print(f"  Warning: '{DISTRACTED_LABEL}' not in class names, "
              f"using index 1 as Distracted")

    model = MLP(
        in_dim=len(feature_cols),
        hidden_dims=bundle["hidden_dims"],
        num_classes=num_classes,
        dropout=bundle["dropout"],
    ).to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    scaler_mean  = np.asarray(bundle["scaler_mean"],  dtype=np.float64)
    scaler_scale = np.asarray(bundle["scaler_scale"], dtype=np.float64)

    print("\nSetting up MediaPipe ...")
    ensure_models()
    face_lm, pose_lm, hand_lm = make_landmarkers()

    # ── Open camera or video ─────────────────────────────────────────────────
    if VIDEO_FILE:
        print(f"\nOpening video file: {VIDEO_FILE}")
        cap = cv2.VideoCapture(VIDEO_FILE)
    else:
        print(f"\nOpening webcam (index {WEBCAM_INDEX}) ...")
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open video source")
        return

    # ── Run calibration before the main loop ────────────────────────────────
    calibration_baseline = None
    if VIDEO_FILE and SKIP_CALIBRATION_FOR_VIDEO:
        print("Skipping calibration for video file input.")
    else:
        calibration_baseline = run_calibration(
            cap, face_lm, pose_lm, hand_lm,
            duration_s=CALIBRATION_DURATION_S,
            mirror=not VIDEO_FILE,
        )
        if calibration_baseline is None:
            print("Calibration was cancelled or failed; running without it.")

    print("\nControls:")
    print("  Q / ESC  — quit")
    print("  M        — toggle audio mute")
    print("  C        — recalibrate baseline\n")

    # ── State ────────────────────────────────────────────────────────────────
    buffer = deque(maxlen=window_size)

    ema_probs           = np.ones(num_classes) / num_classes  # uniform prior
    displayed_class     = None
    candidate_class     = None
    switch_counter      = 0

    distracted_start_t  = None        # wall time when Distracted first held
    last_severity       = "MONITORING"
    last_severe_beep_t  = 0.0

    audio_muted         = False
    fps_window          = deque(maxlen=30)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break
        if not VIDEO_FILE:
            frame = cv2.flip(frame, 1)

        # 1) per-frame features (calibrated if baseline is available)
        feats = extract_from_frame(frame, face_lm, pose_lm, hand_lm)
        feats = apply_calibration(feats, calibration_baseline)
        buffer.append(feats)

        # 2) prediction with temperature scaling
        if len(buffer) == window_size:
            x_temporal = compute_window_stats(buffer, raw_feature_names)
            x_scaled   = (x_temporal - scaler_mean) / scaler_scale
            with torch.no_grad():
                xt = torch.from_numpy(
                    x_scaled.astype(np.float32)
                ).unsqueeze(0).to(DEVICE)
                logits = model(xt) / TEMPERATURE
                raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 3) EMA smoothing on probabilities (for display + thresholding)
            ema_probs = EMA_ALPHA * raw_probs + (1 - EMA_ALPHA) * ema_probs

            # 4) Hysteresis on the displayed class
            argmax_class = int(np.argmax(ema_probs))
            if displayed_class is None:
                displayed_class = argmax_class
                switch_counter  = 0
            elif argmax_class == displayed_class:
                switch_counter  = 0
            else:
                if argmax_class == candidate_class:
                    switch_counter += 1
                else:
                    candidate_class = argmax_class
                    switch_counter  = 1
                if switch_counter >= MIN_SWITCH_FRAMES:
                    displayed_class = argmax_class
                    switch_counter  = 0

        # 5) Severity tracking
        now = time.time()
        if displayed_class == distracted_idx:
            if distracted_start_t is None:
                distracted_start_t = now
            distracted_duration = now - distracted_start_t
        else:
            distracted_start_t  = None
            distracted_duration = 0.0

        severity = classify_severity(distracted_duration)

        # 6) Audio alerts (only on transitions or while in SEVERE)
        if not audio_muted:
            if severity != last_severity:
                if severity == "DISTRACTED":
                    beep_async(*DISTRACTED_BEEP)
                elif severity == "SEVERE":
                    beep_async(*SEVERE_BEEP)
                    last_severe_beep_t = now
            elif severity == "SEVERE":
                if (now - last_severe_beep_t) >= SEVERE_BEEP_INTERVAL_S:
                    beep_async(*SEVERE_BEEP)
                    last_severe_beep_t = now
        last_severity = severity

        # 7) FPS
        dt = time.time() - t0
        fps_window.append(dt)
        avg_fps = 1.0 / np.mean(fps_window) if fps_window else 0.0

        # 8) UI
        draw_overlay(
            frame, class_names, displayed_class, ema_probs,
            severity, distracted_duration,
            len(buffer), avg_fps, window_size, audio_muted,
        )
        cv2.imshow("Driver Distraction Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('m'), ord('M')):
            audio_muted = not audio_muted
            print(f"  Audio {'muted' if audio_muted else 'unmuted'}")
        if key in (ord('c'), ord('C')) and not VIDEO_FILE:
            new_baseline = run_calibration(
                cap, face_lm, pose_lm, hand_lm,
                duration_s=CALIBRATION_DURATION_S, mirror=True,
            )
            if new_baseline is not None:
                calibration_baseline = new_baseline
                # Reset state so post-calibration predictions start fresh
                buffer.clear()
                ema_probs          = np.ones(num_classes) / num_classes
                displayed_class    = None
                candidate_class    = None
                switch_counter     = 0
                distracted_start_t = None
                last_severity      = "MONITORING"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()