"""
Real-time Face Mask Detection on Live Webcam / Video Feed using OpenCV and CNN (TensorFlow/Keras).
Usage:
    python detect_mask_video.py                  # Live webcam (default device index 0)
    python detect_mask_video.py --camera 1       # Use external webcam index 1
    python detect_mask_video.py --video test.mp4 # Run on a video file
    python detect_mask_video.py --output out.avi # Save output annotated video
Press 'q' or 'ESC' to exit.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_mask_detector():
    """Finds and loads the trained .h5 model."""
    candidate_paths = [
        "mask_detector.h5",
        "mask_detector.model.h5",
        os.path.join("face-mask-detection", "mask_detector.h5"),
        os.path.join("face-mask-detection", "mask_detector.model.h5"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"[INFO] Loading trained model from: {path}")
            return load_model(path)
    raise FileNotFoundError("Could not find mask_detector.h5 model file. Please train first.")


def draw_styled_label(frame, text, x, y, color, bg_color=None):
    """Draws a clean text label with an optional solid background banner for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Calculate box coordinates
    margin = 4
    box_y1 = max(0, y - text_h - 2 * margin)
    box_y2 = y
    box_x1 = x
    box_x2 = x + text_w + 2 * margin
    
    # Draw background rectangle
    if bg_color is None:
        bg_color = (0, 0, 0)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    # Draw text
    cv2.putText(frame, text, (box_x1 + margin, box_y2 - margin), font, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Live Face Mask Detection")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("-v", "--video", type=str, default="", help="Path to optional input video file")
    parser.add_argument("-o", "--output", type=str, default="", help="Path to optional output annotated video file")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    # Load model
    model = load_mask_detector()

    # Load OpenCV's pre-trained Haar Cascade Face Detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print(f"[ERROR] Haar cascade XML not found at {cascade_path}")
        sys.exit(1)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Initialize video capture (webcam or file)
    video_source = args.video if args.video else args.camera
    source_name = f"video file '{args.video}'" if args.video else f"webcam (index {args.camera})"
    print(f"[INFO] Opening {source_name}...")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {source_name}.")
        if not args.video:
            print("[TIP] If no physical webcam is connected, test on a video or image using:")
            print("      python detect_mask_video.py --video <path_to_video>")
            print("      python detect_mask_image.py <path_to_image>")
        sys.exit(1)

    # Optional video writer setup
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"[INFO] Writing output to: {args.output}")

    print("[INFO] Starting real-time detection. Press 'q' or 'ESC' in the window to quit.")

    prev_time = time.time()
    fps_history = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Reached end of video stream or failed to grab frame.")
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-5)
        prev_time = current_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        (h_frame, w_frame) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        mask_count = 0
        no_mask_count = 0

        # Process each detected face
        for (x, y, w, h) in faces:
            # Safe ROI bounding
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_frame, x + w), min(h_frame, y + h)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                continue

            # Convert BGR to RGB and resize to MobileNetV2 input size (224, 224)
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_array = np.expand_dims(preprocess_input(face_resized.astype("float32")), axis=0)

            # Predict mask vs no_mask
            preds = model.predict(face_array, verbose=0)[0]
            # Classes are ['with_mask', 'without_mask'] -> index 0: mask, index 1: without_mask
            mask_prob, no_mask_prob = preds[0], preds[1]

            if mask_prob > no_mask_prob:
                label = f"Mask: {mask_prob * 100:.1f}%"
                color = (0, 220, 0)      # Green
                mask_count += 1
            else:
                label = f"No Mask: {no_mask_prob * 100:.1f}%"
                color = (0, 0, 230)      # Red
                no_mask_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw corner accents for high-tech HUD look
            line_len = int(min(w, h) * 0.2)
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, 4)

            # Draw text label banner above face
            draw_styled_label(frame, label, x1, y1 - 6, color, bg_color=(20, 20, 20))

        # Draw HUD bar at top
        hud_text = f"FPS: {avg_fps:.1f} | Faces: {len(faces)} | Mask: {mask_count} | No Mask: {no_mask_count}"
        draw_styled_label(frame, hud_text, 10, 30, (255, 255, 255), bg_color=(30, 30, 30))

        if writer is not None:
            writer.write(frame)

        # Show live display window
        try:
            cv2.imshow("Face Mask Detection (Press 'q' to Quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
        except Exception:
            # In headless environments without display
            pass

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stream ended.")


if __name__ == "__main__":
    main()
