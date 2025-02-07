import streamlit as st

# ✅ Move this to the top before any other Streamlit command
st.set_page_config(page_title="Tennis Analysis App")

import os
import time
import tempfile
import shutil
import cv2
import gdown  # Google Drive downloader
from PyDrive2.auth import GoogleAuth
from PyDrive2.drive import GoogleDrive
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

# ✅ Google Drive Folder ID to Store Results
GDRIVE_RESULTS_FOLDER = "your_google_drive_folder_id"

# ✅ Authenticate Google Drive Access
def authenticate_google_drive():
    """Authenticate with Google Drive."""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Use this for local authentication
    return GoogleDrive(gauth)

drive = authenticate_google_drive()

# Directory to store models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

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
if "processed_video_url" not in st.session_state:
    st.session_state.processed_video_url = None
if "heatmap_image_url" not in st.session_state:
    st.session_state.heatmap_image_url = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Upload video file
uploaded_file = st.file_uploader("📂 Upload a Tennis Match Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    input_video_path = os.path.join(temp_dir, uploaded_file.name)
    temp_output_video = os.path.join(temp_dir, "processed_video.mp4")
    temp_heatmap_image = os.path.join(temp_dir, "heatmap.jpg")

    ball_hits_csv = os.path.join(temp_dir, "ball_hits_coordinates.csv")
    transformed_csv = os.path.join(temp_dir, "transformed_ball_hits_coordinates.csv")

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

            # ✅ Upload files to Google Drive
            def upload_to_drive(file_path, file_name):
                try:
                    file_drive = drive.CreateFile({
                        "title": file_name,
                        "parents": [{"id": GDRIVE_RESULTS_FOLDER}]
                    })
                    file_drive.SetContentFile(file_path)
                    file_drive.Upload()
                    return f"https://drive.google.com/uc?id={file_drive['id']}"
                except Exception as e:
                    st.error(f"❌ Google Drive Upload Error: {e}")
                    return None

            processed_video_url = upload_to_drive(temp_output_video, "processed_video.mp4")
            heatmap_image_url = upload_to_drive(temp_heatmap_image, "heatmap.jpg")

            # ✅ Assign URLs to session state
            if processed_video_url:
                st.session_state.processed_video_url = processed_video_url
            if heatmap_image_url:
                st.session_state.heatmap_image_url = heatmap_image_url
            st.session_state.processing_done = True

# ✅ Display Results from Google Drive
if st.session_state.processing_done:
    st.subheader("🎬 Processed Video")

    if st.session_state.processed_video_url:
        st.video(st.session_state.processed_video_url)
    else:
        st.error("❌ Error: Processed video could not be displayed.")

    st.subheader("📊 Heatmap of Ball Hits")

    if st.session_state.heatmap_image_url:
        st.image(st.session_state.heatmap_image_url, use_container_width=True)
    else:
        st.error("❌ Heatmap file missing.")

    # ✅ Debugging Output
    st.write("📂 Debugging Information:")
    st.write(f"Processed Video URL: {st.session_state.processed_video_url}")
    st.write(f"Heatmap Image URL: {st.session_state.heatmap_image_url}")

    # ✅ Download buttons
    st.write("📥 Download Processed Files:")

    if st.session_state.processed_video_url:
        st.markdown(f"[⬇ Download Processed Video]({st.session_state.processed_video_url})")

    if st.session_state.heatmap_image_url:
        st.markdown(f"[⬇ Download Heatmap]({st.session_state.heatmap_image_url})")
