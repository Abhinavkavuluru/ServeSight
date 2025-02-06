from Court_Detector.video_utils import read_video
from Court_Detector.court_line_detector import CourtLineDetector

class VideoProcessor:
    def __init__(self, input_video_path, court_model_path):
        self.input_video_path = input_video_path
        self.court_model_path = court_model_path
        self.video_frames = None
        self.court_line_detector = None
        self.court_keypoints = None

    def read_video(self):
        self.video_frames = read_video(self.input_video_path)

    def load_model(self):
        self.court_line_detector = CourtLineDetector(self.court_model_path)

    def detect_court_lines(self):
        if self.video_frames is not None:
            self.court_keypoints = self.court_line_detector.predict(self.video_frames[0])
            self.court_line_detector.save_keypoints_to_csv(self.court_keypoints)
        else:
            print("Video frames not loaded. Please call read_video() first.")

    def run(self):
        self.read_video()
        self.load_model()
        self.detect_court_lines()
        print("Court coordinates are saved into csv.")

