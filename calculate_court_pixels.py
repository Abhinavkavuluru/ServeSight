import cv2
import numpy as np
import pandas as pd

def calculate_pixels_based_on_coordinates(coords_csv):
    # Load original court key points from the coordinates CSV file
    coords_df = pd.read_csv(coords_csv, index_col=0)
    coords_pts = coords_df.loc[[0, 1, 2, 3], ["X", "Y"]].values.astype(np.float32)

    # Calculate court width and height based on the original court points
    court_width = coords_pts[3][0] - coords_pts[2][0]  # Updated court width
    court_height = coords_pts[2][1] - coords_pts[0][1]  # Updated court height

    # Calculate the middle y-coordinate for top_left and top_right points
    middle_y_top_left = coords_pts[0][1] / 2
    middle_y_top_right = coords_pts[1][1] / 2

    # Update the y-coordinates for top_left and top_right points
    cropped_court_pts = np.array([
        [0, middle_y_top_left],  # Top Left → (0, middle_y_top_left)
        [court_width, middle_y_top_right],  # Top Right → (court_width, middle_y_top_right)
        [0, court_height],  # Bottom Left → (0, court_height)
        [court_width, court_height]  # Bottom Right → (court_width, court_height)
    ], dtype=np.float32)

    # Compute the homography matrix
    H, _ = cv2.findHomography(coords_pts, cropped_court_pts)

    # Define the region of interest based on the calculated edge coordinates
    roi_corners = np.array([
        [0, middle_y_top_left],
        [court_width, middle_y_top_right],
        [court_width, court_height],
        [0, court_height]
    ], dtype=np.int32)

    # Create a black image with the same dimensions as the court
    image = np.zeros((int(court_height), int(court_width)), dtype=np.uint8)
    # Fill the region of interest with white
    cv2.fillConvexPoly(image, roi_corners, 1)
    # Get the coordinates of the white pixels within the region of interest
    y_coords, x_coords = np.where(image == 1)

    return court_width, court_height
