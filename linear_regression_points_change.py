import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from calculate_court_pixels import calculate_pixels_based_on_coordinates

def calculate_plain_image_dimensions():
    # Coordinates of the court edges in the plain image. These are not changable fixed
    bottom_left = np.array([19, 554])
    bottom_right = np.array([276, 534])
    top_left = np.array([19, 19])
    top_right = np.array([276, 19])

    # Calculate width and height
    width = np.linalg.norm(bottom_right - bottom_left)
    height = np.linalg.norm(top_left - bottom_left)

    return width, height

class CoordinateTransform:
    def __init__(self, input_csv, output_csv, coords_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.coords_csv = coords_csv

    def change_coordinates(self):
        court_width, court_height = calculate_pixels_based_on_coordinates(self.coords_csv)
        plain_width, plain_height = calculate_plain_image_dimensions()

        data = pd.read_csv(self.input_csv)

        # Linear regression models for x and y coordinates
        x_model = LinearRegression()
        y_model = LinearRegression()

        # Create court and plain dimension arrays for fitting
        court_dimensions = np.array([[0, 0], [court_width, court_height]])
        plain_dimensions = np.array([[0, 0], [plain_width, plain_height]])

        # Fit the models
        x_model.fit(court_dimensions[:, 0].reshape(-1, 1), plain_dimensions[:, 0])
        y_model.fit(court_dimensions[:, 1].reshape(-1, 1), plain_dimensions[:, 1])

        # Apply the scaling factors to the x and y coordinates
        data['x'] = data['x'] * x_model.coef_[0] + x_model.intercept_
        data['y'] = data['y'] * y_model.coef_[0] + y_model.intercept_

        # Create a new DataFrame with frame_id, x, and y
        transformed_data = data[['frame_id', 'x', 'y']]

        # Save the new coordinates to a new CSV file with the specific format
        transformed_data.to_csv(self.output_csv, index=False, float_format='%.15f')
        print('Transformed ball hits coordinates have been calculated and saved to the CSV file')
