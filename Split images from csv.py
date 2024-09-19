import os
import pandas as pd
import shutil

# Define paths
csv_file = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test.csv'  # Replace with the actual path to your CSV file
source_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/images'  # Replace with the path to the folder containing the images
destination_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test_images'  # Replace with the path to your destination folder

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Iterate through the image names in the CSV file and copy them to the destination folder
for image_name in df['Images']:
    source_path = os.path.join(source_folder, image_name)
    destination_path = os.path.join(destination_folder, image_name)

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied {image_name} to {destination_folder}")
    else:
        print(f"File {image_name} not found in {source_folder}")

print("Image extraction completed.")
