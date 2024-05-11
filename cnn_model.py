import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

!unzip TRAIN.zip
!unzip TEST.zip

train_folder = 'TRAIN'
test_folder = 'TEST'
labels_file = 'labels.csv'

#Store labels into DataFrame
labels_df = pd.read_csv(labels_file)

def preprocess_images(folder_path, target_size):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array /= 255.0
            images.append(img_array)
    return np.array(images)

train_images = preprocess_images(train_folder, target_size=(200, 200))
test_images = preprocess_images(test_folder, target_size=(200, 200))

#Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

#Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

#CNN Model
model = Sequential([
    Conv2D(16, (3, 3), activation='elu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='elu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='elu'),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(128, activation='elu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with augmented data
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Data metrics
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')