import streamlit as st
import os
import time
import tempfile
import cv2
import gdown
from dotline import DotLine
from ball_hits import BallTracker
from heatmap import TennisHeatmap

# ✅ Fix OpenCV VideoWriter encoder issue for Streamlit Cloud
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ✅ Initialize session state variables
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "heatmap_image" not in st.session_state:
    st.session_state.heatmap_image = None

# ✅ Google Drive model file IDs
GDRIVE_FILES = {
    "yolo5_last.pt": "1YegZe9_HXEVuXEA-dbjn70DbBv0vxbFR",
    "ball_tracker.pkl": "1tjM6IVFVf-q5-fcWryC3H1ytlfNbcNeR"
}

# ✅ Ensure model directory exists
MODEL_DIR = os.path.abspath("models")
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(file_name, file_id):
    """Download required models from Google Drive if missing."""
    file_path = os.path.join(MODEL_DIR, file_name)

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with st.spinner(f"📥 Downloading {file_name} from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        st.error(f"❌ Error: {file_name} download failed. Check file permissions.")
        return None

    return file_path

# ✅ Ensure YOLO model and stub exist
MODEL_PATH = download_file("yolo5_last.pt", GDRIVE_FILES["yolo5_last.pt"])
STUB_PATH = download_file("ball_tracker.pkl", GDRIVE_FILES["ball_tracker.pkl"])

# ✅ Streamlit UI
st.sidebar.title("📋 How to Use")
st.title("🎾 Tennis Match Analysis App")
st.write("Upload a tennis match video to process and generate a heatmap.")

uploaded_file = st.file_uploader("📂 Upload a Tennis Match Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()

    input_video_path = os.path.join(temp_dir, uploaded_file.name)
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")
    heatmap_image = os.path.join(temp_dir, "heatmap.jpg")
    ball_hits_csv = os.path.join(temp_dir, "ball_hits_coordinates.csv")
    transformed_csv = os.path.join(temp_dir, "transformed_ball_hits_coordinates.csv")

    # ✅ Save uploaded file
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # ✅ Debugging
    if not os.path.exists(input_video_path):
        st.error("❌ Error: Uploaded video not saved correctly.")
    else:
        st.subheader("🎥 Uploaded Video")
        st.video(input_video_path)

    # ✅ Process Video
    if st.button("⚡ Process Video & Generate Heatmap"):
        st.write("⏳ Processing video, please wait...")

        with st.spinner("🔄 Processing video..."):
            tracker = DotLine(MODEL_PATH, input_video_path, output_video_path)
            tracker.process_video()

        with st.spinner("📌 Tracking ball hits..."):
            hits = BallTracker(MODEL_PATH, input_video_path, STUB_PATH, ball_hits_csv)
            hits.process_ball_hits()

        with st.spinner("🌡️ Generating heatmap..."):
            heatmap = TennisHeatmap(transformed_csv, heatmap_image)
            heatmap.generate_heatmap()

        # ✅ Debugging: Check if files exist
        time.sleep(2)
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            st.session_state.processed_video = output_video_path
            st.success(f"✅ Processed video saved: {output_video_path}")
        else:
            st.error("❌ Processed video is empty.")

        if os.path.exists(heatmap_image) and os.path.getsize(heatmap_image) > 0:
            st.session_state.heatmap_image = heatmap_image
            st.success(f"✅ Heatmap generated: {heatmap_image}")
        else:
            st.error("❌ Heatmap image is empty.")

        st.session_state.processing_done = True

# ✅ Fix OpenCV Encoding Issue
def convert_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        st.error("❌ Error: Unable to read processed video.")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ✅ Use compatible encoding
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        st.error("❌ Error: Processed video has invalid dimensions.")
        return False

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return True

# ✅ Display Results
if st.session_state.processing_done:
    st.subheader("🎬 Processed Video")

    if st.session_state.processed_video:
        converted_video_path = os.path.join(temp_dir, "converted_video.mp4")
        success = convert_video(st.session_state.processed_video, converted_video_path)

        if success and os.path.exists(converted_video_path):
            st.video(converted_video_path)
        else:
            st.error("❌ Error: Processed video could not be displayed.")

    st.subheader("📊 Heatmap of Ball Hits")
    if st.session_state.heatmap_image:
        st.image(st.session_state.heatmap_image, use_column_width=True)
    else:
        st.error("❌ Heatmap file missing.")

    # ✅ Debugging Output
    st.write("📂 Debugging Information:")
    st.write(f"Processed Video Path: {st.session_state.processed_video}")
    st.write(f"Heatmap Image Path: {st.session_state.heatmap_image}")
