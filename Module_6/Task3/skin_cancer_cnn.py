import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load metadata
data_dir = "data"
metadata = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))

# Map image paths
img_dir1 = os.path.join(data_dir, "HAM10000_images_part_1")
img_dir2 = os.path.join(data_dir, "HAM10000_images_part_2")
image_paths = {
    os.path.splitext(f)[0]: os.path.join(img_dir1, f) for f in os.listdir(img_dir1)
}
image_paths.update(
    {os.path.splitext(f)[0]: os.path.join(img_dir2, f) for f in os.listdir(img_dir2)}
)

metadata["path"] = metadata["image_id"].map(image_paths)

# Simplify labels: Benign vs Malignant
binary_mapping = {
    "nv": "benign",
    "bkl": "benign",
    "df": "benign",
    "vasc": "benign",
    "mel": "malignant",
    "bcc": "malignant",
    "akiec": "malignant",
}
metadata["label"] = metadata["dx"].map(binary_mapping)

# Image preprocessing
IMG_SIZE = 128
images, labels = [], []

for _, row in metadata.iterrows():
    img = cv2.imread(row["path"])
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(row["label"])

X = np.array(images) / 255.0
le = LabelEncoder()
y = le.fit_transform(labels)
y_cat = to_categorical(y, num_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
)

# CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=10,
)

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion matrix
sns.heatmap(
    confusion_matrix(y_true, y_pred),
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the model
os.makedirs("model", exist_ok=True)
model.save("model/skin_cancer_cnn_model.h5")
