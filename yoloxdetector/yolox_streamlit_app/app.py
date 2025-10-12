import streamlit as st
from yolox_inference import run_image_inference, run_video_inference
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="YOLOX Object Detection", layout="wide")
st.title("YOLOX Object Detection App")

st.sidebar.header("Detection Settings")

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
nms_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.45, 0.05)

input_size = st.sidebar.selectbox(
    "Test Input Size (Width × Height)",
    options=[(640, 640), (896, 896), (1024, 1024), (1024, 1536), (1536, 1536), (2048, 2048), (3072, 3072)],
    index=1
)

show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_scores = st.sidebar.checkbox("Show Confidence Scores", value=True)
show_box = st.sidebar.checkbox("Draw Bounding Boxes", value=True)

# Load class names from exp config
from yolox_inference import exp
selected_class = st.sidebar.selectbox(
    "Only show detections for class",
    options=["All"] + exp.class_names,
    index=0
)

choice = st.radio("Select Input Type", ("Image", "Video"))

if choice == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            output_path = run_image_inference(img, conf_thresh, nms_thresh, input_size,
                                              show_box, show_labels, show_scores, selected_class)
            if output_path:
                st.image(Image.open(output_path), caption="Detection Result", use_column_width=True)
            else:
                st.warning("No detections found.")

elif choice == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            temp_path = tmp.name
        st.video(temp_path)

        if st.button("Run Detection"):
            progress = st.empty()
            output_path = run_video_inference(temp_path, conf_thresh, nms_thresh, input_size,
                                              show_box, show_labels, show_scores, selected_class, progress)

            if output_path and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                    st.video(video_bytes)
            else:
                st.error("Detection failed or output file not found.")
