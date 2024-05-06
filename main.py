import os
import cv2
import numpy as np
import pandas as pd

# Function to load images from a folder and resize them
def load_images(folder, target_size):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, target_size)  # Resize image
                images.append(img)
    return images

# Function to load labels from a CSV file
def load_labels(csv_file):
    label_map = {}
    data = pd.read_csv(csv_file)
    for index, row in data.iterrows():
        label_map[row['Image']] = row['Category']
    return label_map

# directory containing images
images_folder = '/Users/ricky_nguyen12/Desktop/images'
# Labels file
labels_file = 'labels.csv'
# directory containing training images
train_images_folder = '/Users/ricky_nguyen12/Desktop/TRAIN'
# directory containing test images
test_images_folder = '/Users/ricky_nguyen12/Desktop/TEST'

# target size
target_size = (200, 200)

# Load labels from the CSV file
label_map = load_labels(labels_file)

# Load training images and labels
train_images = load_images(train_images_folder, target_size)
train_labels = []
for image_file in os.listdir(train_images_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
        if image_file in label_map:
            train_labels.append(label_map[image_file])

# Load test images and labels
test_images = load_images(test_images_folder, target_size)
test_labels = []
for image_file in os.listdir(test_images_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
        if image_file in label_map:
            test_labels.append(label_map[image_file])

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

