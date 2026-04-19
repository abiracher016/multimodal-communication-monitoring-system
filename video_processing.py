"""
Video Processing Module
- Face detection (OpenCV Haar cascades)
- Facial emotion recognition (DeepFace)
- Attention estimation (MediaPipe Face Mesh — iris/gaze)
- Head movement / posture stability
"""
import os
import traceback
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from backend.config import (
    VIDEO_ANALYSIS_FPS, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS,
    ATTENTION_THRESHOLD,
)


# ─── Face Detection ──────────────────────────────────────────────────────────
def detect_faces(frame, face_cascade):
    """Detect faces in a single frame. Returns list of (x,y,w,h) rects."""
    if frame is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS
    )
    return faces if len(faces) > 0 else []


# ─── Emotion Recognition ─────────────────────────────────────────────────────
def analyze_emotion(frame) -> dict:
    """
    Analyze facial emotion using DeepFace.
    Returns { dominant_emotion: str, emotions: dict }
    """
    try:
        from deepface import DeepFace
        results = DeepFace.analyze(
            frame, actions=["emotion"], enforce_detection=False, silent=True
        )
        if isinstance(results, list):
            results = results[0]
        return {
            "dominant_emotion": results.get("dominant_emotion", "unknown"),
            "emotions": results.get("emotion", {}),
        }
    except Exception:
        return {"dominant_emotion": "unknown", "emotions": {}}


# ─── Attention Estimation (MediaPipe) ─────────────────────────────────────────
def estimate_attention(frame, face_mesh) -> dict:
    """
    Estimate attention/gaze direction using MediaPipe Face Mesh.
    Returns { looking_at_screen: bool, attention_score: float, head_pose: str }
    """
    try:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"looking_at_screen": False, "attention_score": 0.0, "head_pose": "no_face"}

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Nose tip (landmark 1) for head pose estimation
        nose = landmarks.landmark[1]
        nose_x_normalized = nose.x  # 0-1, 0.5 = center

        # Left iris center (landmarks 468-472) and right iris (473-477)
        # Using simplified approach with eye landmarks
        left_eye_inner = landmarks.landmark[133]
        left_eye_outer = landmarks.landmark[33]
        right_eye_inner = landmarks.landmark[362]
        right_eye_outer = landmarks.landmark[263]

        # Check horizontal gaze — if nose is roughly centered, person is looking at screen
        center_deviation = abs(nose_x_normalized - 0.5)

        # Vertical check
        nose_y = nose.y
        vertical_deviation = abs(nose_y - 0.45)

        # Combined attention score
        horizontal_score = max(0, 1.0 - center_deviation * 3)
        vertical_score = max(0, 1.0 - vertical_deviation * 3)
        attention_score = round((horizontal_score * 0.7 + vertical_score * 0.3), 2)

        looking_at_screen = attention_score >= ATTENTION_THRESHOLD

        # Head pose
        if nose_x_normalized < 0.35:
            head_pose = "looking_left"
        elif nose_x_normalized > 0.65:
            head_pose = "looking_right"
        elif nose_y < 0.3:
            head_pose = "looking_up"
        elif nose_y > 0.6:
            head_pose = "looking_down"
        else:
            head_pose = "facing_camera"

        return {
            "looking_at_screen": looking_at_screen,
            "attention_score": attention_score,
            "head_pose": head_pose,
        }

    except Exception as e:
        return {"looking_at_screen": False, "attention_score": 0.0, "head_pose": "error"}


# ─── Head Movement / Stability ────────────────────────────────────────────────
def calculate_head_stability(pose_history: list[dict]) -> float:
    """
    Calculate head stability from pose history.
    Lower movement = higher stability score.
    """
    if len(pose_history) < 2:
        return 100.0

    movements = 0
    for i in range(1, len(pose_history)):
        if pose_history[i].get("head_pose") != pose_history[i - 1].get("head_pose"):
            movements += 1

    stability = max(0, 100 - (movements / len(pose_history)) * 200)
    return round(stability, 2)


# ─── Full Video Pipeline ─────────────────────────────────────────────────────
def process_video(video_path: str) -> dict:
    """
    Full video analysis pipeline.
    Returns aggregate metrics for the entire video.
    """
    if cv2 is None:
        return _empty_video_result("OpenCV not installed")

    if not os.path.exists(video_path):
        return _empty_video_result(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _empty_video_result(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # How many frames to skip between analyses
    frame_skip = max(1, int(fps / VIDEO_ANALYSIS_FPS))

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Initialize MediaPipe Face Mesh
    face_mesh = None
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    except ImportError:
        print("[Video] MediaPipe not available, skipping attention analysis")

    # Per-frame results
    frame_results = []
    emotion_counts = {}
    face_count_total = 0
    attention_scores = []
    pose_history = []
    frames_analyzed = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        frames_analyzed += 1
        result = {"frame": frame_idx, "time": round(frame_idx / fps, 2) if fps > 0 else 0}

        # Face detection
        faces = detect_faces(frame, face_cascade)
        result["face_count"] = len(faces)
        face_count_total += len(faces)

        # Emotion (on first detected face)
        if len(faces) > 0:
            x, y, w, h = faces[0] if isinstance(faces, np.ndarray) else faces[0]
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size > 0:
                emotion_data = analyze_emotion(face_roi)
                result["emotion"] = emotion_data["dominant_emotion"]
                emotion_counts[emotion_data["dominant_emotion"]] = \
                    emotion_counts.get(emotion_data["dominant_emotion"], 0) + 1

        # Attention (MediaPipe)
        if face_mesh is not None:
            attention_data = estimate_attention(frame, face_mesh)
            result["attention"] = attention_data
            attention_scores.append(attention_data["attention_score"])
            pose_history.append(attention_data)

        frame_results.append(result)
        frame_idx += 1

    cap.release()
    if face_mesh:
        face_mesh.close()

    # Aggregate metrics
    avg_faces = round(face_count_total / frames_analyzed, 2) if frames_analyzed > 0 else 0
    avg_attention = round(sum(attention_scores) / len(attention_scores), 2) if attention_scores else 0
    attention_pct = round(
        sum(1 for s in attention_scores if s >= ATTENTION_THRESHOLD) / len(attention_scores) * 100, 2
    ) if attention_scores else 0

    head_stability = calculate_head_stability(pose_history)

    # Dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"

    # Emotion distribution (percentages)
    total_emotion_frames = sum(emotion_counts.values()) or 1
    emotion_distribution = {
        k: round(v / total_emotion_frames * 100, 2)
        for k, v in emotion_counts.items()
    }

    return {
        "status": "success",
        "duration": round(duration, 2),
        "frames_analyzed": frames_analyzed,
        "avg_faces_detected": avg_faces,
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "avg_attention_score": avg_attention,
        "attention_percentage": attention_pct,
        "head_stability": head_stability,
        "frame_results": frame_results[:100],  # limit for API response size
    }


def _empty_video_result(reason: str) -> dict:
    """Return empty result when video processing not possible."""
    return {
        "status": "unavailable",
        "reason": reason,
        "duration": 0,
        "frames_analyzed": 0,
        "avg_faces_detected": 0,
        "dominant_emotion": "unknown",
        "emotion_distribution": {},
        "avg_attention_score": 0,
        "attention_percentage": 0,
        "head_stability": 0,
        "frame_results": [],
    }
