import cv2
import numpy as np
import pandas as pd

class Homography:
    def __init__(self, input_csv, output_csv, coords_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.coords_csv = coords_csv

        # Read original court key points from the coordinates CSV file
        self.original_court_pts = self.load_coordinates()

        # Calculate court width and height based on the original court points
        self.court_width = self.original_court_pts[3][0] - self.original_court_pts[2][0]  # Updated court width
        self.court_height = self.original_court_pts[2][1] - self.original_court_pts[0][1]  # Updated court height

        # Calculate the middle y-coordinate for top_left and top_right points
        middle_y_top_left = self.original_court_pts[0][1] / 2
        middle_y_top_right = self.original_court_pts[1][1] / 2

        # Update the y-coordinates for top_left and top_right points
        self.cropped_court_pts = np.array([
            [0, middle_y_top_left],  # Top Left → (0, middle_y_top_left)
            [self.court_width, middle_y_top_right],  # Top Right → (court_width, middle_y_top_right)
            [0, self.court_height],  # Bottom Left → (0, court_height)
            [self.court_width, self.court_height]  # Bottom Right → (court_width, court_height)
        ], dtype=np.float32)

        # Compute the homography matrix
        self.H, _ = cv2.findHomography(self.original_court_pts, self.cropped_court_pts)

    def load_coordinates(self):
        coords_df = pd.read_csv(self.coords_csv)
        # Map coordinates from the CSV file based on index values
        coords_pts = coords_df.loc[[0, 1, 2, 3], ["X", "Y"]].values.astype(np.float32)
        return coords_pts

    def transform_coordinates(self):
        df = pd.read_csv(self.input_csv)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")  # Convert to float, NaN if invalid
        df["y"] = pd.to_numeric(df["y"], errors="coerce")  # Convert to float, NaN if invalid
        df = df.dropna(subset=["x", "y"])  # Drop rows where x or y is NaN

        transformed_data = []
        for _, row in df.iterrows():
            frame_id = int(row["frame_id"])  # Ensure frame ID is an integer
            x, y = float(row["x"]), float(row["y"])

            ball_center = np.array([[[x, y]]], dtype=np.float32)
            transformed_center = cv2.perspectiveTransform(ball_center, self.H)
            new_x, new_y = transformed_center[0][0]

            transformed_data.append([frame_id, new_x, new_y])

        transformed_df = pd.DataFrame(transformed_data, columns=["frame_id", "x", "y"])
        transformed_df.to_csv(self.output_csv, index=False)
        return self.output_csv
