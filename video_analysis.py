import cv2

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    face_count = 0

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_count += len(faces)

    cap.release()

    avg_faces = face_count / frame_count if frame_count else 0

    return {
        "avg_faces_detected": round(avg_faces, 2)
    }