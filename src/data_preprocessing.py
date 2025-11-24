import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATASET_DIR = "../dataset"          # Path to dataset folder
IMG_SIZE = 224                      # Image size for CNN
CATEGORIES = ["with_mask", "without_mask"]


# -------------------------------------------------
# LOAD IMAGES FROM FOLDERS
# -------------------------------------------------
def load_dataset():
    data = []
    labels = []

    for label, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_DIR, category)

        for img_name in os.listdir(category_path):

            img_path = os.path.join(category_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip corrupted files

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            labels.append(label)

    data = np.array(data) / 255.0  # Normalize
    labels = np.array(labels)

    return data, labels


# -------------------------------------------------
# CREATE DATA GENERATORS WITH AUGMENTATION
# -------------------------------------------------
def create_generators(X_train, X_test, y_train, y_test):
    
    train_aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_aug = ImageDataGenerator()

    train_gen = train_aug.flow(
        X_train, y_train, batch_size=32
    )

    test_gen = test_aug.flow(
        X_test, y_test, batch_size=32
    )

    return train_gen, test_gen


# -------------------------------------------------
# VISUALIZE SAMPLE IMAGES
# -------------------------------------------------
def visualize_samples(X, y):
    plt.figure(figsize=(6, 6))
    for i in range(9):
        idx = np.random.randint(0, len(X))
        plt.subplot(3, 3, i+1)
        plt.imshow(X[idx])
        plt.title("With Mask" if y[idx] == 0 else "Without Mask")
        plt.axis("off")
    plt.show()


# -------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()

    print("Dataset loaded successfully!")
    print("Total images:", len(X))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # Create data generators
    train_gen, test_gen = create_generators(X_train, X_test, y_train, y_test)

    print("Data generators created!")

    # Visualize few samples
    visualize_samples(X_train, y_train)

    print("Preprocessing complete. Ready for model training.")
