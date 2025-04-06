import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(data_path, img_size=(150, 150)):
    """
    Load and preprocess image data from directories.
    Expected structure:
    - data_path/
        - with_mask/
            - img1.jpg, img2.jpg, ...
        - without_mask/
            - img1.jpg, img2.jpg, ...
    """
    print(f"Loading data from {data_path}...")

    images = []
    labels = []

    # Load images with masks
    with_mask_path = os.path.join(data_path, 'with_mask')
    if os.path.exists(with_mask_path):
        for img_name in os.listdir(with_mask_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(with_mask_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append('with_mask')
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # Load images without masks
    without_mask_path = os.path.join(data_path, 'without_mask')
    if os.path.exists(without_mask_path):
        for img_name in os.listdir(without_mask_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(without_mask_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append('without_mask')
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # Convert to numpy arrays
    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=labels)

    print(f"Data loaded: {len(images)} images")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    return (X_train, y_train), (X_test, y_test), le.classes_

def preprocess_image(image_path, img_size=(150, 150)):
    """
    Preprocess a single image for prediction
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = np.array(img, dtype="float32") / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None
