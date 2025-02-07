import streamlit as st
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

# Google Drive file IDs
GDRIVE_FILES = {
    "yolo5_last.pt": "1YegZe9_HXEVuXEA-dbjn70DbBv0vxbFR",
    "ball_tracker.pkl": "1tjM6IVFVf-q5-fcWryC3H1ytlfNbcNeR"
}

# ✅ Persistent storage directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to download files from Google Drive
def download_file(file_name, file_id):
    file_path = os.path.join("models", file_name)
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading {file_name} from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    return file_path

# ✅ Download required models from Google Drive
MODEL_PATH = download_file("yolo5_last.pt", GDRIVE_FILES["yolo5_last.pt"])
STUB_PATH = download_file("ball_tracker.pkl", GDRIVE_FILES["ball_tracker.pkl"])

# Sidebar Instructions
st.sidebar.title("📋 How to Use")
st.sidebar.markdown("""
- **Upload a Tennis Video** 🎾 (MP4, AVI, MOV)  
- **Preview the Video** after uploading  
- **Process the Video**  
- **Generate & Download the Heatmap** 🔥  
""")

# App Title
st.title("🎾 Tennis Match Analysis App")
st.write("Upload a tennis match video to process and generate a heatmap.")

# Initialize session state
for key in ["processed_video", "heatmap_image", "processing_done"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Upload video file
uploaded_file = st.file_uploader("📂 Upload a Tennis Match Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    input_video_path = os.path.join(temp_dir, uploaded_file.name)
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")
    converted_video_path = os.path.join(temp_dir, "converted_video.mp4")

    heatmap_image = os.path.join(temp_dir, "heatmap.jpg")
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
                tracker = DotLine(MODEL_PATH, input_video_path, output_video_path)
                tracker.process_video()

            # Step 2: Track ball hits
            with st.spinner("📌 Tracking ball hits..."):
                hits = BallTracker(MODEL_PATH, input_video_path, STUB_PATH, ball_hits_csv)
                hits.process_ball_hits()

            # Step 3: Generate heatmap
            with st.spinner("🌡️ Generating heatmap..."):
                heatmap = TennisHeatmap(transformed_csv, heatmap_image)
                heatmap.generate_heatmap()

            # ✅ Move results to persistent storage
            final_video_path = os.path.join(OUTPUT_DIR, "processed_video.mp4")
            final_heatmap_path = os.path.join(OUTPUT_DIR, "heatmap.jpg")

            if os.path.exists(output_video_path):
                shutil.move(output_video_path, final_video_path)
                st.session_state.processed_video = final_video_path

            if os.path.exists(heatmap_image):
                shutil.move(heatmap_image, final_heatmap_path)
                st.session_state.heatmap_image = final_heatmap_path

            # ✅ Ensure session state updates properly
            st.session_state.processing_done = True

# ✅ Display results only after processing
if st.session_state.processing_done:
    st.subheader("🎬 Processed Video")
    
    if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
        st.video(st.session_state.processed_video)
    else:
        st.error("❌ Processed video missing.")

    st.subheader("📊 Heatmap of Ball Hits")

    if st.session_state.heatmap_image and os.path.exists(st.session_state.heatmap_image):
        st.image(st.session_state.heatmap_image, use_column_width=True)
    else:
        st.error("❌ Heatmap file missing.")

    # ✅ Debugging Output
    st.write("📂 Debugging Information:")
    st.write(f"Processed Video Path: {st.session_state.processed_video if st.session_state.processed_video else '❌ Missing'}")
    st.write(f"Heatmap Image Path: {st.session_state.heatmap_image if st.session_state.heatmap_image else '❌ Missing'}")

    # ✅ Download buttons
    st.write("📥 Download Processed Files:")
    
    if st.session_state.heatmap_image:
        with open(st.session_state.heatmap_image, "rb") as file:
            st.download_button("⬇ Download Heatmap", data=file, file_name="heatmap.jpg")

    if st.session_state.processed_video:
        with open(st.session_state.processed_video, "rb") as file:
            st.download_button("⬇ Download Processed Video", data=file, file_name="processed_video.mp4")
