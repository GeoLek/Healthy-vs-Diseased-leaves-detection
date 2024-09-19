import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        print(f"GPU Available: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU.")

# Load CSV files
train_csv_file = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/train.csv'  # Update with the actual path to your training CSV file
test_csv_file = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test.csv'  # Update with the actual path to your testing CSV file

# Read CSV files
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

# Debugging: Print the actual column names of the DataFrame
print("Train CSV Columns:", train_df.columns)
print("Test CSV Columns:", test_df.columns)

# Convert Labels to strings for binary classification in the training data
train_df['Labels'] = train_df['Labels'].astype(str)

# Define paths
train_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/train_images'  # Update with your training images folder path
test_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test_images'  # Update with your test images folder path

# Parameters
img_width, img_height = 2048, 1365  # Resize the images to 2048x1365 pixels
batch_size = 8  # Adjusted batch size to fit within 16GB VRAM
epochs = 10

# Create ImageDataGenerators for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the data for training
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_folder,
    x_col='Images',
    y_col='Labels',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the model using VGG19
vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
vgg_base.trainable = False  # Freeze the VGG19 layers to prevent overfitting

model = models.Sequential([
    vgg_base,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Added dropout for regularization
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=train_generator,
    validation_steps=train_generator.samples // batch_size
)

# Save the model
model.save('leaf_disease_model_vgg19.h5')

# Load the trained model (useful if you restart the session)
model = load_model('leaf_disease_model_vgg19.h5')

# Prepare the test data (no labels needed)
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

# Predict on the test data
predictions = model.predict(test_generator, steps=test_generator.samples)

# Convert predictions to binary labels
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Create a DataFrame with image names and predicted labels
results_df = pd.DataFrame({
    'Images': test_generator.filenames,
    'Labels': predicted_labels
})

# Save the results to a new CSV file
results_csv_path = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/results_vgg19.csv'  # Update with your desired output CSV file path
results_df.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")
