import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Dropout


# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        print(f"GPU Available: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU.")

# Load CSV files
train_csv_file = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/train.csv'
test_csv_file = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test.csv'

# Read CSV files
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

# Debugging: Print the actual column names of the DataFrame
print("Train CSV Columns:", train_df.columns)
print("Test CSV Columns:", test_df.columns)

# Convert Labels to strings for binary classification in the training data
train_df['Labels'] = train_df['Labels'].astype(str)

# Define paths
train_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/train_images'
test_folder = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/test_images'

# Parameters
img_width, img_height = 240, 160  # Resize the images to 240x160 pixels for faster training
batch_size = 8
epochs = 50

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


# Enhanced model
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 4
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 5
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # Fully Connected Layers
    layers.Dense(1024, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
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
model.save('leaf_disease_model.h5')

# Load the trained model (useful if you restart the session)
model = load_model('leaf_disease_model.h5')

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
results_csv_path = '/home/orion/Geo/Kaggle competitions/D4-Computer Vision/results_(another copy).csv'
results_df.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")
