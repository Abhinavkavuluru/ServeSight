import streamlit as st
import os
import time
import tempfile
import cv2
from dotline import DotLine
from ball_hits import BallTracker
from heatmap import TennisHeatmap
from image_ploting import ImagePlotter


# Set up Streamlit page configuration
st.set_page_config(page_title="Tennis Analysis App")

# Sidebar with instructions
st.sidebar.title("📋 How to Use")
st.sidebar.markdown(
    """
    - **Upload a Tennis Video** 🎾 (MP4, AVI, MOV)  
    - **Preview the Video** after uploading  
    - **View Ball Tracking Data** 📊  
    - **Check the Heatmap** 🔥  
    - **Download the Heatmap**  
    """
)

# Streamlit App Title
st.title("🎾 Tennis Match Analysis App")
st.write("Upload a tennis match video to process, track ball movements, and generate a heatmap.")

# Ensure output directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Paths for model and processing
MODEL_PATH = os.path.join(BASE_DIR, "yolo5_last.pt")
STUB_PATH = os.path.join(BASE_DIR, "ball_tracker.pkl")

# Initialize session state
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "heatmap_image" not in st.session_state:
    st.session_state.heatmap_image = None
if "output_image" not in st.session_state:
    st.session_state.output_image = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Upload video file
uploaded_file = st.file_uploader("📂 Upload a Tennis Match Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    input_video_path = os.path.join(temp_dir, uploaded_file.name)
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")
    converted_video_path = os.path.join(temp_dir, "converted_video.mp4")

    heatmap_image = os.path.join(OUTPUT_DIR, "heatmap.jpg")
    output_image = os.path.join(OUTPUT_DIR, "court_plot.jpg")
    ball_hits_csv = os.path.join(OUTPUT_DIR, "ball_hits_coordinates.csv")
    transformed_csv = os.path.join(OUTPUT_DIR, "transformed_ball_hits_coordinates.csv")

    # Save uploaded file
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display uploaded video
    st.subheader("🎥 Uploaded Video")
    st.video(input_video_path)

    # Processing button
    if st.button("⚡ Process Video & Generate Heatmap"):
        st.write("⏳ Processing video, please wait...")

        # Step 1: Process the video (from video.py)
        with st.spinner("🔄 Processing video..."):
            tracker = DotLine(MODEL_PATH, input_video_path, output_video_path)
            tracker.process_video()

        # Step 2: Track ball hits and generate coordinates
        with st.spinner("📌 Tracking ball hits..."):
            hits = BallTracker(MODEL_PATH, input_video_path, STUB_PATH, ball_hits_csv)
            hits.process_ball_hits()

        # Step 3: Generate heatmap
        with st.spinner("🌡️ Generating heatmap..."):
            heatmap = TennisHeatmap(transformed_csv, heatmap_image)
            heatmap.generate_heatmap()

        # Step 4: Plot ball hits on the court
        with st.spinner("📍 Plotting ball hits on the court..."):
            plotter = ImagePlotter(transformed_csv, input_video_path, output_image)
            plotter.plot_coordinates_on_image()

        # Verify files
        time.sleep(2)

        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            st.session_state.processed_video = output_video_path
            st.success("✅ Video processing complete!")
        else:
            st.error("❌ Processed video is missing.")

        if os.path.exists(heatmap_image) and os.path.getsize(heatmap_image) > 0:
            st.session_state.heatmap_image = heatmap_image
            st.success("✅ Heatmap generated successfully!")
        else:
            st.error("❌ Heatmap missing!")

        if os.path.exists(output_image) and os.path.getsize(output_image) > 0:
            st.session_state.output_image = output_image
            st.success("✅ Court plot generated successfully!")
        else:
            st.error("❌ Court plot missing!")

        st.session_state.processing_done = True

# Display outputs only if processing is done
if st.session_state.processing_done:
    st.subheader("🎬 Processed Video")

    if st.session_state.processed_video:
        # Convert video to a display-compatible format
        def convert_video(input_path, output_path):
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
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

        success = convert_video(st.session_state.processed_video, converted_video_path)

        if success:
            with open(converted_video_path, "rb") as video_file:
                processed_video_bytes = video_file.read()
                if len(processed_video_bytes) > 0:
                    st.video(processed_video_bytes)
                else:
                    st.error("❌ Converted video is empty.")
        else:
            st.error("❌ Could not convert processed video.")

    # Display Heatmap
    st.subheader("📊 Heatmap of Ball Hits")
    if st.session_state.heatmap_image:
        with open(st.session_state.heatmap_image, "rb") as img_file:
            img_bytes = img_file.read()
            st.image(img_bytes, use_column_width=True)
    else:
        st.error("⚠️ Heatmap image not found.")

    # Display Ball Hits on Court
    st.subheader("📌 Ball Hits on the Court")
    if st.session_state.output_image:
        with open(st.session_state.output_image, "rb") as img_file:
            img_bytes = img_file.read()
            st.image(img_bytes, use_column_width=True)
    else:
        st.error("⚠️ Court plot image not found.")

    # Download buttons
    st.write("📥 Download Processed Files:")

    if st.session_state.heatmap_image:
        with open(st.session_state.heatmap_image, "rb") as file:
            st.download_button("⬇ Download Heatmap", data=file, file_name="heatmap.jpg")

    if st.session_state.output_image:
        with open(st.session_state.output_image, "rb") as file:
            st.download_button("⬇ Download Court Plot", data=file, file_name="court_plot.jpg")
