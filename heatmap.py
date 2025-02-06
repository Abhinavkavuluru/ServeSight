import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class TennisHeatmap:
    def __init__(self, direction_changes_csv, output_heatmap_path):
        self.direction_changes_csv = direction_changes_csv
        self.output_heatmap_path = output_heatmap_path

    def generate_heatmap(self):
        # ✅ Check if the CSV file exists before proceeding
        if not os.path.exists(self.direction_changes_csv):
            print(f"❌ ERROR: CSV file not found - {self.direction_changes_csv}")
            return

        print(f"📂 Loading CSV file: {self.direction_changes_csv}")

        # Read data
        try:
            data = pd.read_csv(self.direction_changes_csv)
        except Exception as e:
            print(f"❌ ERROR: Failed to read CSV file. Exception: {e}")
            return

        if data.empty:
            print("⚠️ WARNING: CSV file is empty. No heatmap will be generated.")
            return

        # Check if necessary columns exist
        if 'x' not in data.columns or 'y' not in data.columns:
            print("❌ ERROR: CSV file is missing 'x' or 'y' columns.")
            return

        # Generate heatmap
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=data['x'], y=data['y'], cmap='hot', fill=True)

        # ✅ Ensure the directory exists
        os.makedirs(os.path.dirname(self.output_heatmap_path), exist_ok=True)

        # ✅ Save the heatmap
        plt.savefig(self.output_heatmap_path)
        plt.close()

        print(f"✅ Heatmap saved successfully at: {self.output_heatmap_path}")
