import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class TennisHeatmap:
    def __init__(self, direction_changes_csv, output_heatmap, heatmap_width=295, heatmap_height=551):
        self.direction_changes_csv = direction_changes_csv
        self.output_heatmap = output_heatmap
        self.heatmap_width = heatmap_width
        self.heatmap_height = heatmap_height
        self.court_points = [
            (19, 19), (19, 534), (276, 19), (276, 534),
            (44, 19), (44, 534), (251, 20), (251, 534),
            (19, 276), (276, 276), (147, 135), (146, 420),
            (19, 19), (276, 19), (19, 534), (276, 534),
            (45, 133), (251, 133), (44, 422), (251, 422)
        ]
        self.court_lines = [
            (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
            (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)
        ]
        self.colormap_dict = {
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "OCEAN": cv2.COLORMAP_OCEAN,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "INFERNO": cv2.COLORMAP_INFERNO
        }

    def generate_heatmap(self, selected_colormap="OCEAN"):
        # ✅ Ensure the CSV file exists
        if not os.path.exists(self.direction_changes_csv):
            print(f"❌ ERROR: CSV file not found - {self.direction_changes_csv}")
            return

        print(f"📂 Loading CSV file: {self.direction_changes_csv}")

        # Read data safely
        try:
            data = pd.read_csv(self.direction_changes_csv)
        except Exception as e:
            print(f"❌ ERROR: Failed to read CSV file. Exception: {e}")
            return

        # ✅ Check if CSV has the correct columns
        if 'x' not in data.columns or 'y' not in data.columns:
            print("❌ ERROR: CSV file is missing 'x' or 'y' columns. Cannot generate heatmap.")
            return

        # ✅ Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_heatmap), exist_ok=True)

        # Create a blank heatmap array
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)

        # Populate heatmap with direction change points
        for _, row in data.iterrows():
            try:
                x, y = int(row['x']), int(row['y'])
                if 0 <= x < self.heatmap_width and 0 <= y < self.heatmap_height:
                    heatmap[y, x] += 1  # Increment intensity at this point
            except ValueError:
                print(f"⚠️ WARNING: Skipped invalid row with x={row.get('x')} and y={row.get('y')}")

        # Apply Gaussian blur to smooth the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

        # Normalize heatmap for visualization
        heatmap_normalized = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        # Convert to a color map
        colormap = self.colormap_dict.get(selected_colormap, cv2.COLORMAP_OCEAN)

        for line in self.court_lines:
            pt1 = self.court_points[line[0]]
            pt2 = self.court_points[line[1]]
            cv2.line(heatmap_normalized, pt1, pt2, (255), 2)  # White lines on grayscale heatmap

        # Apply the colormap after drawing the court lines
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)

        # Matplotlib color bar setup
        fig, ax = plt.subplots(figsize=(18, 10))
        im = ax.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))  # Display heatmap
        plt.title("Ball Hits Intensity")

        # Adjust the color bar to match the height of the court
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(selected_colormap.lower()), norm=plt.Normalize(vmin=0, vmax=np.max(heatmap)))
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label('Intensity (Frequency of Direction Changes)', fontsize=12)

        # ✅ Save the final image safely
        plt.savefig(self.output_heatmap, bbox_inches='tight', dpi=200)
        print(f"✅ Heatmap saved successfully at: {self.output_heatmap}")
