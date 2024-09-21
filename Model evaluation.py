import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        print(f"GPU Available: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU.")

# Load CSV file
test_csv_file = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test.csv'  # Update with the actual path to your testing CSV file
test_df = pd.read_csv(test_csv_file)

# Define paths
test_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test_images'  # Update with your test images folder path

# Parameters
img_width, img_height = 1024, 683  # Resize the images to 1024x683 pixels
batch_size = 8  # Adjusted batch size to fit within 16GB VRAM

# Create ImageDataGenerator for rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Prepare the test data (no labels needed for prediction)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_folder,
    x_col='Images',
    y_col=None,  # No labels for prediction
    target_size=(img_width, img_height),
    batch_size=1,  # Predict one image at a time
    class_mode=None,  # No class_mode since we're predicting
    shuffle=False
)

# Load the trained model
model = load_model('leaf_disease_model_vgg19.h5')

# Predict on the test data
predictions = model.predict(test_generator, steps=test_generator.samples)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Create a DataFrame with image names and predicted labels
results_df = pd.DataFrame({
    'Images': test_generator.filenames,
    'Predicted_Labels': predicted_labels
})

# Save the results to a new CSV file
results_csv_path = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/results_vgg19.csv'  # Update with your desired output CSV file path
results_df.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")
