import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Import preprocessing functions
from data_preprocessing import load_dataset

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INIT_LR = 1e-4     # learning rate
EPOCHS = 10
BS = 32            # batch size
MODEL_PATH = "../models/mask_detector.h5"


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
print("[INFO] Loading dataset...")
X, y = load_dataset()

print("[INFO] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# -------------------------------------------------
# LOAD BASE MODEL (MobileNetV2)
# -------------------------------------------------
print("[INFO] Loading MobileNetV2 base model...")

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False


# -------------------------------------------------
# BUILD HEAD MODEL
# -------------------------------------------------
print("[INFO] Building head model...")

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)


# -------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
print("[INFO] Training model...")

H = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    batch_size=BS,
    epochs=EPOCHS
)


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
print("[INFO] Saving model...")
os.makedirs("../models", exist_ok=True)
model.save(MODEL_PATH)

print(f"[INFO] Model saved to {MODEL_PATH}")
