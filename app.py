import streamlit as st

# ✅ Move this to the top before any other Streamlit command
st.set_page_config(page_title="Tennis Analysis App")

import os
import time
import tempfile
import shutil
import cv2
import gdown  # Google Drive file downloader
from dotline import DotLine
from ball_hits import BallTracker
from heatmap import TennisHeatmap

# ✅ Fix OpenCV VideoWriter encoder issue
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Google Drive file IDs (Replace with actual IDs)
GDRIVE_FILES = {
    "yolo5_last.pt": "1YegZe9_HXEVuXEA-dbjn70DbBv0vxbFR",
    "ball_tracker.pkl": "1tjM6IVFVf-q5-fcWryC3H1ytlfNbcNeR"
}

# Directory to store models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Persistent storage for processed files
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to download files from Google Drive
def download_file(file_name, file_id):
    """Download model files from Google Drive and ensure directory exists."""
    file_path = os.path.join(MODEL_DIR, file_name)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download only if missing
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with st.spinner(f"Downloading {file_name} from Google Drive..."):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            st.error(f"❌ ERROR: Failed to download {file_name}. Please check your Google Drive link.")
            return None

        return file_path
    except Exception as e:
        st.error(f"❌ ERROR: Could not download {file_name}. Exception: {e}")
        return None

# ✅ Download required models from Google Drive
MODEL_PATH = download_file("yolo5_last.pt", GDRIVE_FILES["yolo5_last.pt"])
STUB_PATH = download_file("ball_tracker.pkl", GDRIVE_FILES["ball_tracker.pkl"])

if not MODEL_PATH or not STUB_PATH:
    st.error("❌ ERROR: Required model files are missing. Please check the logs for details.")
    st.stop()

# Sidebar with instructions
st.sidebar.title("📋 How to Use")
st.sidebar.markdown(
    """
    - **Upload a Tennis Video** 🎾 (MP4, AVI, MOV)  
    - **Preview the Video** after uploading  
    - **Process the Video**  
    - **Generate & Download the Heatmap** 🔥  
    """
)

# Streamlit App Title
st.title("🎾 Tennis Match Analysis App")
st.write("Upload a tennis match video to process and generate a heatmap.")

# Initialize session state
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "heatmap_image" not in st.session_state:
    st.session_state.heatmap_image = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Upload video file
uploaded_file = st.file_uploader("📂 Upload a Tennis Match Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    input_video_path = os.path.join(temp_dir, uploaded_file.name)
    temp_output_video = os.path.join(temp_dir, "processed_video.mp4")
    final_output_video = os.path.join(OUTPUT_DIR, "processed_video.mp4")

    temp_heatmap_image = os.path.join(temp_dir, "heatmap.jpg")
    final_heatmap_image = os.path.join(OUTPUT_DIR, "heatmap.jpg")

    ball_hits_csv = os.path.join(OUTPUT_DIR, "ball_hits_coordinates.csv")
    transformed_csv = os.path.join(OUTPUT_DIR, "transformed_ball_hits_coordinates.csv")

    # Save uploaded file
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # ✅ Check if video file is valid before processing
    if not os.path.exists(input_video_path) or os.path.getsize(input_video_path) == 0:
        st.error("❌ Error: Uploaded video is empty or invalid.")
    else:
        # Display uploaded video
        st.subheader("🎥 Uploaded Video")
        st.video(input_video_path)

        # Processing button
        if st.button("⚡ Process Video & Generate Heatmap"):
            st.write("⏳ Processing video, please wait...")

            # Step 1: Process the video
            with st.spinner("🔄 Processing video..."):
                tracker = DotLine(MODEL_PATH, input_video_path, temp_output_video)
                tracker.process_video()

            # Step 2: Track ball hits and generate coordinates
            with st.spinner("📌 Tracking ball hits..."):
                hits = BallTracker(MODEL_PATH, input_video_path, STUB_PATH, ball_hits_csv)
                hits.process_ball_hits()

            # Step 3: Generate heatmap
            with st.spinner("🌡️ Generating heatmap..."):
                heatmap = TennisHeatmap(transformed_csv, temp_heatmap_image)
                heatmap.generate_heatmap()

            # ✅ Move processed files to persistent storage
            shutil.move(temp_output_video, final_output_video)
            shutil.move(temp_heatmap_image, final_heatmap_image)

            # ✅ Assign paths to session state
            st.session_state.processed_video = final_output_video
            st.session_state.heatmap_image = final_heatmap_image
            st.session_state.processing_done = True

# ✅ Fix for missing converted video issue
if st.session_state.processing_done:
    st.subheader("🎬 Processed Video")

    if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
        st.video(st.session_state.processed_video)
    else:
        st.error("❌ Error: Processed video could not be displayed.")

    st.subheader("📊 Heatmap of Ball Hits")

    if st.session_state.heatmap_image and os.path.exists(st.session_state.heatmap_image):
        st.image(st.session_state.heatmap_image, use_container_width=True)  # ✅ Fix for deprecated `use_column_width`
    else:
        st.error("❌ Heatmap file missing.")

    # ✅ Debugging Output
    st.write("📂 Debugging Information:")
    st.write(f"Processed Video Path: {st.session_state.processed_video}")
    st.write(f"Heatmap Image Path: {st.session_state.heatmap_image}")

    # ✅ Download buttons
    st.write("📥 Download Processed Files:")

    if st.session_state.processed_video:
        with open(st.session_state.processed_video, "rb") as file:
            st.download_button("⬇ Download Processed Video", data=file, file_name="processed_video.mp4")

    if st.session_state.heatmap_image:
        with open(st.session_state.heatmap_image, "rb") as file:
            st.download_button("⬇ Download Heatmap", data=file, file_name="heatmap.jpg")
