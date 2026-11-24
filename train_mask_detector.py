"""
Train a Face Mask Classifier using MobileNetV2 Transfer Learning and TensorFlow/Keras.
Dataset layout:
    dataset/with_mask/*.jpg
    dataset/without_mask/*.jpg

Outputs:
    - mask_detector.h5 (and mask_detector.model.h5)
    - plot.png (Training loss and accuracy curves)
"""

import os
import sys

# Ensure local cache for Keras pretrained weights is used
workspace_dir = os.path.dirname(os.path.abspath(__file__))
keras_home = os.path.join(workspace_dir, ".keras")
if not os.path.exists(keras_home):
    parent_keras = os.path.join(os.path.dirname(workspace_dir), ".keras")
    if os.path.exists(parent_keras):
        keras_home = parent_keras
os.environ["KERAS_HOME"] = keras_home
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Configuration
INIT_LR = 1e-4
EPOCHS = 15
BS = 32
IMAGE_SIZE = (224, 224)
CATEGORIES = ["with_mask", "without_mask"]

# Find dataset directory
DATASET = "dataset"
if not os.path.exists(DATASET) and os.path.exists(os.path.join("face-mask-detection", "dataset")):
    DATASET = os.path.join("face-mask-detection", "dataset")

print(f"[INFO] Using dataset directory: {DATASET}")
print("[INFO] Loading and preprocessing images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATASET, category)
    if not os.path.exists(path):
        print(f"[ERROR] Category path not found: {path}")
        sys.exit(1)
        
    images = os.listdir(path)
    print(f"  -> Loading {len(images)} images for category '{category}'...")
    
    for img_name in images:
        img_path = os.path.join(path, img_name)
        try:
            image = load_img(img_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)
        except Exception as e:
            continue

print(f"[INFO] Total valid images loaded: {len(data)}")

# Convert to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels_bin = lb.fit_transform(labels)
# For binary classification (with_mask, without_mask), lb.classes_ is ['with_mask', 'without_mask']
labels_cat = to_categorical(labels_bin)
print(f"[INFO] Class mappings: {lb.classes_[0]} -> [1, 0], {lb.classes_[1]} -> [0, 1]")

# Train/Test Split (80% train, 20% test)
(trainX, testX, trainY, testY) = train_test_split(
    data, labels_cat, test_size=0.20, stratify=labels_cat, random_state=42
)
print(f"[INFO] Training samples: {len(trainX)}, Testing samples: {len(testX)}")

# Data Augmentation generator
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Build CNN using MobileNetV2 base + custom classification head
print("[INFO] Constructing CNN architecture (MobileNetV2 transfer learning)...")
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model layers so they are not updated during training
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# Train the model
print(f"[INFO] Training CNN classifier for {EPOCHS} epochs with batch size {BS}...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("\n[INFO] Evaluating network performance on test set...")
predIdxs = model.predict(testX, batch_size=BS)
predClasses = np.argmax(predIdxs, axis=1)
trueClasses = np.argmax(testY, axis=1)

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(trueClasses, predClasses, target_names=lb.classes_))

# Save the model
output_path_h5 = "mask_detector.h5"
output_path_model_h5 = "mask_detector.model.h5"
print(f"[INFO] Saving model weights to '{output_path_h5}' and '{output_path_model_h5}'...")
model.save(output_path_h5)
model.save(output_path_model_h5)

# Also copy to face-mask-detection subfolder if it exists
subfolder = "face-mask-detection"
if os.path.exists(subfolder):
    model.save(os.path.join(subfolder, output_path_h5))
    model.save(os.path.join(subfolder, output_path_model_h5))

print("[INFO] Model saved successfully.")

# Plot training loss and accuracy
plot_path = "plot.png"
print(f"[INFO] Saving training loss and accuracy curve to '{plot_path}'...")
plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="Train Loss", linewidth=2)
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="Val Loss", linewidth=2)
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.title("Face Mask Detection: Training Loss & Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
if os.path.exists(subfolder):
    plt.savefig(os.path.join(subfolder, plot_path), dpi=300)
print("[INFO] Training complete!")
