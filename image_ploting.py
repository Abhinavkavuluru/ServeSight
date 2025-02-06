import os
import cv2
import pandas as pd

class ImagePlotter:
    def __init__(self, output_csv, image_path, output_image_path):
        self.output_csv = output_csv
        self.image_path = image_path
        self.output_image_path = output_image_path

    def plot_coordinates_on_image(self):
        # ✅ Ensure output directory exists
        output_dir = os.path.dirname(self.output_image_path)
        os.makedirs(output_dir, exist_ok=True)

        # ✅ Check if CSV file exists
        if not os.path.exists(self.output_csv):
            print(f"❌ ERROR: CSV file not found - {self.output_csv}")
            return

        # ✅ Load CSV safely
        try:
            data = pd.read_csv(self.output_csv)
        except Exception as e:
            print(f"❌ ERROR: Failed to read CSV file. Exception: {e}")
            return

        # ✅ Ensure CSV has required columns
        if 'x' not in data.columns or 'y' not in data.columns:
            print("❌ ERROR: CSV file missing 'x' or 'y' columns.")
            return

        # ✅ Check if image exists before loading
        if not os.path.exists(self.image_path):
            print(f"❌ ERROR: Image file not found - {self.image_path}")
            return

        # ✅ Load image safely
        image = cv2.imread(self.image_path)

        if image is None or image.size == 0:
            print(f"❌ ERROR: Failed to load image - {self.image_path}")
            return

        # ✅ Plot coordinates on image
        for _, row in data.iterrows():
            x, y = int(row['x']), int(row['y'])
            cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # Red dot

        # ✅ Save image safely
        success = cv2.imwrite(self.output_image_path, image)

        if success:
            print(f"✅ Image with plotted coordinates saved to: {self.output_image_path}")
        else:
            print(f"❌ ERROR: Failed to save {self.output_image_path}")
