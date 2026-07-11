# ─── extract_features.py (Tasks API) ──────────────────────────────────────────
# One-shot pose/face/hand feature extractor for the driver-distraction dataset.
#
# Uses MediaPipe's modern Tasks API (face_landmarker + pose_landmarker +
# hand_landmarker) instead of the legacy `solutions` API, which is broken
# on recent mediapipe wheels for Python 3.12 / Windows.
#
# For each input image, runs three landmarkers and computes ~22 interpretable
# features capturing WHAT THE PERSON IS DOING (head pose, gaze, mouth, hands,
# posture) rather than WHAT THEY LOOK LIKE.
#
# The .task model files auto-download from Google Storage on first run
# into MODELS_DIR. Combined size: ~25 MB.
#
# Output: a single CSV at OUT_PATH, one row per image, with columns
#   image_path, subject, label, face_detected, pitch, yaw, roll, ...
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import urllib.request

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ────────────────────────────────────────────────────────────────────
LOSO_DIR     = r"C:\Users\Alex Clare\Downloads\Dataset\loso_csvs"
OUT_PATH     = r"C:\Users\Alex Clare\Downloads\Dataset\features_v1.csv"
MODELS_DIR   = r"C:\Users\Alex Clare\Downloads\Dataset\mp_models"
SAMPLE_EVERY = 20      # 1 = no subsampling; 20 ≈ 15K images, ~30-45 min on CPU

FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")

MODEL_URLS = {
    FACE_MODEL: ("https://storage.googleapis.com/mediapipe-models/"
                 "face_landmarker/face_landmarker/float16/latest/"
                 "face_landmarker.task"),
    POSE_MODEL: ("https://storage.googleapis.com/mediapipe-models/"
                 "pose_landmarker/pose_landmarker_lite/float16/latest/"
                 "pose_landmarker_lite.task"),
    HAND_MODEL: ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/"
                 "hand_landmarker.task"),
}


# ── Generic 3D face model for solvePnP head pose (mm from nose tip) ─────────
MODEL_POINTS = np.array([
    (  0.0,   0.0,   0.0),    # nose tip          (1)
    (  0.0, -63.6, -12.5),    # chin              (152)
    (-43.3,  32.7, -26.0),    # left eye outer    (33)
    ( 43.3,  32.7, -26.0),    # right eye outer   (263)
    (-28.9, -28.9, -24.1),    # left mouth corner (61)
    ( 28.9, -28.9, -24.1),    # right mouth corner(291)
], dtype=np.float64)

# Face Mesh landmark indices (Tasks API model emits 478 landmarks: 468 mesh
# + 10 iris). These indices match the legacy MediaPipe face mesh.
LM = {
    "nose":             1,
    "chin":             152,
    "left_eye_outer":   33,
    "left_eye_inner":   133,
    "right_eye_outer":  263,
    "right_eye_inner":  362,
    "left_eye_top":     159,
    "left_eye_bottom":  145,
    "right_eye_top":    386,
    "right_eye_bottom": 374,
    "left_mouth":       61,
    "right_mouth":      291,
    "upper_lip":        13,
    "lower_lip":        14,
}
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# ──────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────────────────────
def ensure_models():
    """Download .task files on first run."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            name = os.path.basename(path)
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  → {path}  ({size_mb:.1f} MB)")


def make_landmarkers():
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return (
        mp_vision.FaceLandmarker.create_from_options(face_opts),
        mp_vision.PoseLandmarker.create_from_options(pose_opts),
        mp_vision.HandLandmarker.create_from_options(hand_opts),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers — operate on a list of NormalizedLandmark
# ──────────────────────────────────────────────────────────────────────────────
def lm_xy(face_lms, idx, w, h):
    p = face_lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def head_pose_pyr(face_lms, w, h):
    image_points = np.array([
        lm_xy(face_lms, LM["nose"],            w, h),
        lm_xy(face_lms, LM["chin"],            w, h),
        lm_xy(face_lms, LM["left_eye_outer"],  w, h),
        lm_xy(face_lms, LM["right_eye_outer"], w, h),
        lm_xy(face_lms, LM["left_mouth"],      w, h),
        lm_xy(face_lms, LM["right_mouth"],     w, h),
    ], dtype=np.float64)

    fx = fy = float(w)
    cam = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = 0.0
    return pitch, yaw, roll


def eye_aspect_ratio(face_lms, top, bot, outer, inner, w, h):
    t = lm_xy(face_lms, top,   w, h)
    b = lm_xy(face_lms, bot,   w, h)
    o = lm_xy(face_lms, outer, w, h)
    i = lm_xy(face_lms, inner, w, h)
    return float(np.linalg.norm(t - b) / max(np.linalg.norm(o - i), 1e-6))


def iris_offset(face_lms, iris_ids, outer, inner, w, h):
    iris = np.array([
        [face_lms[i].x * w, face_lms[i].y * h] for i in iris_ids
    ]).mean(axis=0)
    o = lm_xy(face_lms, outer, w, h)
    i = lm_xy(face_lms, inner, w, h)
    eye_center = (o + i) / 2
    eye_width  = np.linalg.norm(o - i)
    if eye_width < 1e-6:
        return 0.0, 0.0
    off = (iris - eye_center) / eye_width
    return float(off[0]), float(off[1])


def mouth_open_ratio(face_lms, w, h):
    u  = lm_xy(face_lms, LM["upper_lip"],   w, h)
    l  = lm_xy(face_lms, LM["lower_lip"],   w, h)
    lf = lm_xy(face_lms, LM["left_mouth"],  w, h)
    rt = lm_xy(face_lms, LM["right_mouth"], w, h)
    return float(np.linalg.norm(u - l) / max(np.linalg.norm(lf - rt), 1e-6))


# ──────────────────────────────────────────────────────────────────────────────
# Per-image extraction
# ──────────────────────────────────────────────────────────────────────────────
EMPTY_FEATS = {
    "face_detected": 0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
    "left_ear": 0.0, "right_ear": 0.0,
    "iris_lx": 0.0, "iris_ly": 0.0, "iris_rx": 0.0, "iris_ry": 0.0,
    "mouth_open": 0.0,
    "left_hand_detected": 0, "left_hand_x": 0.0, "left_hand_y": 0.0,
    "left_hand_to_face": 0.0,
    "right_hand_detected": 0, "right_hand_x": 0.0, "right_hand_y": 0.0,
    "right_hand_to_face": 0.0,
    "pose_detected": 0, "shoulder_angle": 0.0,
}


def extract_features(image_path, face_lm, pose_lm, hand_lm):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_res = face_lm.detect(mp_img)
    pose_res = pose_lm.detect(mp_img)
    hand_res = hand_lm.detect(mp_img)

    f = dict(EMPTY_FEATS)
    nose_norm = None

    # ── Face ──────────────────────────────────────────────────────────────
    if face_res.face_landmarks:
        flms = face_res.face_landmarks[0]   # list[NormalizedLandmark]
        f["face_detected"] = 1
        try:
            f["pitch"], f["yaw"], f["roll"] = head_pose_pyr(flms, w, h)
        except Exception:
            pass
        f["left_ear"] = eye_aspect_ratio(
            flms, LM["left_eye_top"], LM["left_eye_bottom"],
            LM["left_eye_outer"], LM["left_eye_inner"], w, h
        )
        f["right_ear"] = eye_aspect_ratio(
            flms, LM["right_eye_top"], LM["right_eye_bottom"],
            LM["right_eye_outer"], LM["right_eye_inner"], w, h
        )
        if len(flms) >= 478:    # iris landmarks present
            f["iris_lx"], f["iris_ly"] = iris_offset(
                flms, LEFT_IRIS,
                LM["left_eye_outer"], LM["left_eye_inner"], w, h
            )
            f["iris_rx"], f["iris_ry"] = iris_offset(
                flms, RIGHT_IRIS,
                LM["right_eye_outer"], LM["right_eye_inner"], w, h
            )
        f["mouth_open"] = mouth_open_ratio(flms, w, h)
        nose_xy = lm_xy(flms, LM["nose"], w, h)
        nose_norm = nose_xy / np.array([w, h])

    # ── Hands ─────────────────────────────────────────────────────────────
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

    # ── Pose ──────────────────────────────────────────────────────────────
    if pose_res.pose_landmarks:
        plms = pose_res.pose_landmarks[0]
        f["pose_detected"] = 1
        ls = plms[11]      # left shoulder
        rs = plms[12]      # right shoulder
        f["shoulder_angle"] = float(np.degrees(
            np.arctan2(ls.y - rs.y, ls.x - rs.x)
        ))

    return f


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def subject_from_path(p):
    m = re.search(r"[\\/](P\d+)[\\/]", str(p))
    if m:
        return m.group(1)
    m = re.search(r"\bP(\d+)\b", str(p))
    return f"P{m.group(1)}" if m else None


def collect_image_paths():
    seen = set()
    rows = []
    for n in range(1, 11):
        for prefix in ("train", "val"):
            csv = os.path.join(LOSO_DIR, f"{prefix}{n}.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            for _, r in df.iterrows():
                p = str(r.iloc[0])
                if p not in seen:
                    seen.add(p)
                    rows.append((p, int(r.iloc[1])))
    return pd.DataFrame(rows, columns=["image_path", "label"])


def main():
    print("Setting up MediaPipe Tasks landmarkers ...")
    ensure_models()
    face_lm, pose_lm, hand_lm = make_landmarkers()
    print("All landmarkers loaded.\n")

    df = collect_image_paths()
    df["subject"] = df["image_path"].apply(subject_from_path)
    print(f"Collected {len(df):,} unique images")
    print("Per subject:", df.groupby("subject").size().to_dict())

    if SAMPLE_EVERY > 1:
        df = df.iloc[::SAMPLE_EVERY].reset_index(drop=True)
        print(f"Subsampled to {len(df):,} images (every {SAMPLE_EVERY}th)\n")

    out_rows = []
    n_failed_read = 0
    n_no_face = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        feats = extract_features(row["image_path"], face_lm, pose_lm, hand_lm)
        if feats is None:
            n_failed_read += 1
            continue
        if feats["face_detected"] == 0:
            n_no_face += 1
        out_rows.append({
            "image_path": row["image_path"],
            "subject":    row["subject"],
            "label":      row["label"],
            **feats,
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out_df):,} rows → {OUT_PATH}")
    print(f"  Failed image reads : {n_failed_read}")
    print(f"  No face detected   : {n_no_face} "
          f"({100 * n_no_face / max(len(out_df), 1):.1f}%)")
    print(f"  Per-subject yield  :")
    print(out_df.groupby("subject").size().to_string())


if __name__ == "__main__":
    main()