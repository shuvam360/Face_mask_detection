"""
Flask Web Application for Real-Time Face Mask Detection.
Features:
- Browser-based Live Webcam Stream with bounding boxes & HUD metrics.
- Photo Upload with instantaneous mask classification & bounding boxes.
- Interactive Dashboard UI.

Run with:
    python web_app.py
Or in production:
    gunicorn web_app:app
"""

import os
import sys
import io
import base64
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Global Model & Cascades
MODEL = None
FACE_CASCADE = None
FACE_CASCADE_ALT = None


def init_ai():
    global MODEL, FACE_CASCADE, FACE_CASCADE_ALT
    # Load Model
    candidate_paths = [
        "mask_detector.h5",
        "mask_detector.model.h5",
        os.path.join("face-mask-detection", "mask_detector.h5")
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            print(f"[INFO] Loading CNN Model from: {p}")
            MODEL = load_model(p)
            break

    # Load Haar Cascades (both default and alt2 for high sensitivity)
    try:
        def_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        alt_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        FACE_CASCADE = cv2.CascadeClassifier(def_path)
        FACE_CASCADE_ALT = cv2.CascadeClassifier(alt_path)
        print("[INFO] AI Models and Face Cascades loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load secondary cascade: {e}")


def draw_styled_label(frame, text, x, y, color, bg_color=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = 4
    box_y1 = max(0, y - text_h - 2 * margin)
    box_y2 = y
    box_x1 = x
    box_x2 = x + text_w + 2 * margin
    if bg_color is None:
        bg_color = (20, 20, 20)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.putText(frame, text, (box_x1 + margin, box_y2 - margin), font, scale, color, thickness, cv2.LINE_AA)


def detect_faces_cascade(gray, w_frame, h_frame):
    """Runs primary and secondary cascade with adaptive thresholds to maximize face recall."""
    faces = ()
    # Pass 1: alt2 cascade (best for real-world faces)
    if FACE_CASCADE_ALT is not None and not FACE_CASCADE_ALT.empty():
        faces = FACE_CASCADE_ALT.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=3,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    # Pass 2: default frontalface cascade fallback
    if len(faces) == 0 and FACE_CASCADE is not None and not FACE_CASCADE.empty():
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    return faces


def process_frame(frame):
    """Processes a frame to detect faces and classify mask status."""
    (h_frame, w_frame) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detect_faces_cascade(gray, w_frame, h_frame)
    faces_info = []

    for (x, y, w, h) in faces:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_frame, int(x + w)), min(h_frame, int(y + h))

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0 or face_roi.shape[0] < 15 or face_roi.shape[1] < 15:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.expand_dims(preprocess_input(face_resized.astype("float32")), axis=0)

        preds = MODEL.predict(face_array, verbose=0)[0]
        mask_prob, no_mask_prob = float(preds[0]), float(preds[1])

        if mask_prob > no_mask_prob:
            label = "Mask"
            confidence = mask_prob
            color = (0, 220, 0)
        else:
            label = "No Mask"
            confidence = no_mask_prob
            color = (0, 0, 230)

        # Draw box and corners
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        line_len = int(min(w, h) * 0.2)
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, 3)

        label_text = f"{label}: {confidence * 100:.1f}%"
        draw_styled_label(frame, label_text, x1, max(24, y1 - 6), color)

        # Strictly native Python data types for clean JSON serialization
        faces_info.append({
            "label": str(label),
            "confidence": float(confidence),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return frame, faces_info


def generate_frames():
    """Server-side video streaming generator function (for local execution)."""
    cap = cv2.VideoCapture(0)
    while True:
        if cap.isOpened():
            success, frame = cap.read()
            if not success:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Webcam feed temporarily unavailable", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            else:
                frame, _ = process_frame(frame)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Use browser camera button below", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        data = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})

        processed_img, faces = process_frame(img)

        # Encode back to base64 for image upload preview
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'faces': faces,
            'image_base64': img_base64
        })
    except Exception as e:
        print(f"[ERROR] predict_image failed: {e}")
        return jsonify({'success': False, 'error': str(e)})


# Automatically initialize AI on module load (required for Gunicorn on Render)
init_ai()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask server on http://localhost:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
