"""
Face Mask Detection on Static Images using OpenCV and CNN (TensorFlow/Keras).
Usage:
    python detect_mask_image.py input.jpg
    python detect_mask_image.py input.jpg --output result.jpg
    python detect_mask_image.py input.jpg --show
"""

import os
import sys
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
    
    margin = 4
    box_y1 = max(0, y - text_h - 2 * margin)
    box_y2 = y
    box_x1 = x
    box_x2 = x + text_w + 2 * margin
    
    if bg_color is None:
        bg_color = (0, 0, 0)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.putText(frame, text, (box_x1 + margin, box_y2 - margin), font, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Image Face Mask Detection")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default="output.jpg", help="Path to save output image (default: output.jpg)")
    parser.add_argument("--show", action="store_true", help="Display image in OpenCV window")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Input image '{args.image}' not found.")
        sys.exit(1)

    # Load model
    model = load_mask_detector()

    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read image
    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Could not decode image at '{args.image}'.")
        sys.exit(1)

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(f"[INFO] Detected {len(faces)} face(s) in image.")

    # Fallback if whole image is already a cropped face (e.g. from dataset)
    if len(faces) == 0:
        print("[INFO] No faces found with standard cascade; evaluating full image as face candidate...")
        faces = [(0, 0, w_img, h_img)]

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        face_roi = img[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.expand_dims(preprocess_input(face_resized.astype("float32")), axis=0)

        preds = model.predict(face_array, verbose=0)[0]
        mask_prob, no_mask_prob = preds[0], preds[1]

        if mask_prob > no_mask_prob:
            label = f"Mask: {mask_prob * 100:.1f}%"
            color = (0, 220, 0)
        else:
            label = f"No Mask: {no_mask_prob * 100:.1f}%"
            color = (0, 0, 230)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Corner accents
        line_len = int(min(w, h) * 0.2)
        cv2.line(img, (x1, y1), (x1 + line_len, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + line_len), color, 3)
        cv2.line(img, (x2, y1), (x2 - line_len, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + line_len), color, 3)
        cv2.line(img, (x1, y2), (x1 + line_len, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - line_len), color, 3)
        cv2.line(img, (x2, y2), (x2 - line_len, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - line_len), color, 3)

        draw_styled_label(img, label, x1, max(24, y1 - 6), color, bg_color=(20, 20, 20))
        print(f"  -> Face at ({x}, {y}, {w}, {h}): {label}")

    cv2.imwrite(args.output, img)
    print(f"[INFO] Annotated result saved to: {args.output}")

    if args.show:
        try:
            cv2.imshow("Detection Result (Press any key to close)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
