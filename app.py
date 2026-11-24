"""
Interactive Streamlit Frontend for Face Mask Detection.
Run with:
    streamlit run app.py
"""

import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Streamlit Page Config
st.set_page_config(
    page_title="Face Mask Detection AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #1E88E5;
    }
    .badge-mask {
        color: #2e7d32;
        background-color: #e8f5e9;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .badge-nomask {
        color: #c62828;
        background-color: #ffebee;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Loads and caches the trained .h5 model."""
    candidate_paths = [
        "mask_detector.h5",
        "mask_detector.model.h5",
        os.path.join("face-mask-detection", "mask_detector.h5"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return load_model(path)
    return None


@st.cache_resource
def get_face_cascade():
    """Loads and caches the OpenCV Haar Cascade classifier."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def draw_styled_label(frame, text, x, y, color, bg_color=None):
    """Draws a clean text label with a solid background banner for clear readability."""
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


def process_image(img_bgr, model, face_cascade, conf_threshold=0.5):
    """Detects faces and predicts mask status for each face in the image."""
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Fallback for cropped single face
    if len(faces) == 0:
        faces = [(0, 0, w_img, h_img)]

    results = []
    output_bgr = img_bgr.copy()

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        face_roi = img_bgr[y1:y2, x1:x2]
        if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.expand_dims(preprocess_input(face_resized.astype("float32")), axis=0)

        preds = model.predict(face_array, verbose=0)[0]
        mask_prob, no_mask_prob = float(preds[0]), float(preds[1])

        is_mask = mask_prob > no_mask_prob and mask_prob >= conf_threshold
        label = "Mask" if is_mask else "No Mask"
        confidence = mask_prob if is_mask else no_mask_prob
        color = (0, 220, 0) if is_mask else (0, 0, 230)

        # Draw bounding box
        cv2.rectangle(output_bgr, (x1, y1), (x2, y2), color, 2)
        # Corner lines
        line_len = int(min(w, h) * 0.2)
        cv2.line(output_bgr, (x1, y1), (x1 + line_len, y1), color, 3)
        cv2.line(output_bgr, (x1, y1), (x1, y1 + line_len), color, 3)
        cv2.line(output_bgr, (x2, y1), (x2 - line_len, y1), color, 3)
        cv2.line(output_bgr, (x2, y1), (x2, y1 + line_len), color, 3)
        cv2.line(output_bgr, (x1, y2), (x1 + line_len, y2), color, 3)
        cv2.line(output_bgr, (x1, y2), (x1, y2 - line_len), color, 3)
        cv2.line(output_bgr, (x2, y2), (x2 - line_len, y2), color, 3)
        cv2.line(output_bgr, (x2, y2), (x2, y2 - line_len), color, 3)

        label_text = f"{label}: {confidence * 100:.1f}%"
        draw_styled_label(output_bgr, label_text, x1, max(24, y1 - 6), color)

        results.append({
            "box": (x1, y1, x2, y2),
            "label": label,
            "confidence": confidence,
            "mask_prob": mask_prob,
            "no_mask_prob": no_mask_prob
        })

    return output_bgr, results


def main():
    st.markdown('<div class="main-header">😷 Face Mask Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Computer Vision using MobileNetV2 CNN, TensorFlow & OpenCV</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/protection-mask.png", width=80)
    st.sidebar.title("Configuration")
    confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    model = get_model()
    face_cascade = get_face_cascade()

    if model is not None:
        st.sidebar.success("✅ CNN Model Loaded (mask_detector.h5)")
    else:
        st.sidebar.error("❌ Model not found! Run train_mask_detector.py first.")
        st.stop()

    if face_cascade is not None:
        st.sidebar.success("✅ OpenCV Face Cascade Loaded")

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Technologies:**
    - MobileNetV2 (Transfer Learning)
    - TensorFlow 2.21 / Keras 3
    - OpenCV 4.14
    - Streamlit
    """)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload Image",
        "📸 Webcam Snapshot",
        "🧪 Sample Gallery",
        "📊 Model Analytics"
    ])

    # TAB 1: Upload Image
    with tab1:
        st.subheader("Upload an Image for Mask Detection")
        uploaded_file = st.file_uploader("Choose a face image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)

            with col2:
                with st.spinner("Detecting faces and classifying masks..."):
                    output_bgr, results = process_image(img_bgr, model, face_cascade, confidence_thresh)
                    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                st.image(output_rgb, caption="Detection Result", use_container_width=True)

            # Summary Metrics
            mask_count = sum(1 for r in results if r["label"] == "Mask")
            nomask_count = sum(1 for r in results if r["label"] == "No Mask")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Faces Detected", len(results))
            m2.metric("Wearing Mask 😷", mask_count)
            m3.metric("No Mask ⚠️", nomask_count)

            # Detailed table
            if results:
                st.markdown("#### Detected Face Details")
                data_rows = []
                for idx, r in enumerate(results, 1):
                    data_rows.append({
                        "Face #": idx,
                        "Status": "😷 Mask" if r["label"] == "Mask" else "❌ No Mask",
                        "Confidence": f"{r['confidence']*100:.2f}%",
                        "Mask Probability": f"{r['mask_prob']*100:.2f}%",
                        "No Mask Probability": f"{r['no_mask_prob']*100:.2f}%",
                        "Bounding Box (x1, y1, x2, y2)": str(r["box"])
                    })
                st.table(data_rows)

                # Download processed image button
                result_pil = Image.fromarray(output_rgb)
                import io
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=byte_im,
                    file_name="annotated_mask_detection.jpg",
                    mime="image/jpeg"
                )

    # TAB 2: Webcam Snapshot
    with tab2:
        st.subheader("Take a Live Photo with Your Webcam")
        st.write("Click below to allow browser camera access and capture a snapshot for real-time mask detection.")
        camera_photo = st.camera_input("Take a photo")

        if camera_photo is not None:
            photo = Image.open(camera_photo).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR)

            output_bgr, results = process_image(img_bgr, model, face_cascade, confidence_thresh)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(photo, caption="Captured Snapshot", use_container_width=True)
            with col2:
                st.image(output_rgb, caption="Mask Classification", use_container_width=True)

            mask_count = sum(1 for r in results if r["label"] == "Mask")
            nomask_count = sum(1 for r in results if r["label"] == "No Mask")
            c1, c2, c3 = st.columns(3)
            c1.metric("Faces Detected", len(results))
            c2.metric("Mask Worn 😷", mask_count)
            c3.metric("No Mask ⚠️", nomask_count)

    # TAB 3: Sample Gallery
    with tab3:
        st.subheader("Test with Pre-loaded Sample Images")
        st.write("Select any sample below to evaluate the model instantly:")

        sample_paths = []
        if os.path.exists("dataset/with_mask"):
            for f in sorted(os.listdir("dataset/with_mask"))[:3]:
                sample_paths.append((os.path.join("dataset/with_mask", f), "With Mask (Sample)"))
        if os.path.exists("dataset/without_mask"):
            for f in sorted(os.listdir("dataset/without_mask"))[:3]:
                sample_paths.append((os.path.join("dataset/without_mask", f), "Without Mask (Sample)"))

        if sample_paths:
            cols = st.columns(len(sample_paths))
            for i, (p, label) in enumerate(sample_paths):
                with cols[i]:
                    st.image(p, caption=label, use_container_width=True)
                    if st.button(f"Analyze #{i+1}", key=f"sample_{i}"):
                        img_bgr = cv2.imread(p)
                        output_bgr, results = process_image(img_bgr, model, face_cascade, confidence_thresh)
                        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                        st.image(output_rgb, caption="Result", use_container_width=True)
                        for r in results:
                            if r["label"] == "Mask":
                                st.success(f"Result: {r['label']} ({r['confidence']*100:.1f}%)")
                            else:
                                st.error(f"Result: {r['label']} ({r['confidence']*100:.1f}%)")
        else:
            st.info("No sample images found in dataset folder.")

    # TAB 4: Model Analytics
    with tab4:
        st.subheader("Model Performance & Architecture")
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("#### Training History")
            if os.path.exists("plot.png"):
                st.image("plot.png", caption="Training & Validation Curves (Loss & Accuracy)", use_container_width=True)
            else:
                st.info("Training plot not found.")

        with col_b:
            st.markdown("#### Test Set Performance Metrics")
            st.markdown("""
            | Metric | With Mask | Without Mask | Macro Avg |
            | :--- | :---: | :---: | :---: |
            | **Precision** | **1.00** | **0.99** | **1.00** |
            | **Recall** | **0.99** | **1.00** | **1.00** |
            | **F1-Score** | **1.00** | **1.00** | **1.00** |
            | **Support** | 138 | 138 | 276 |
            """)

            st.markdown("#### Architecture Specifications")
            st.markdown("""
            - **Base Model:** MobileNetV2 (pre-trained on ImageNet, frozen backbone)
            - **Input Dimensions:** 224 x 224 x 3 (RGB)
            - **Pooling:** AveragePooling2D (7x7)
            - **Dense Layer:** 128 units (ReLU) + Dropout (0.5)
            - **Classification Output:** 2 units (Softmax)
            - **Loss Function:** Binary Crossentropy
            - **Optimizer:** Adam (LR: 1e-4)
            - **Accuracy on Test Split:** **99.64%**
            """)


if __name__ == "__main__":
    main()
