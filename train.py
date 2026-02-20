import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Define dataset path
dataset_path = "my_dataset"

# Step 2: Initialize data and labels
data = []
labels = []

# Step 3: Loop through each folder (A, B, C)
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Read image
        img = cv2.imread(image_path)

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize image
        img = cv2.resize(img, (28,28))

        # Flatten image (convert 2D to 1D)
        img = img.flatten()

        # Append data and label
        data.append(img)
        labels.append(folder)

# Convert into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Step 4: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Step 5: Create SVM model
model = SVC(kernel='linear')

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 8: Save model
joblib.dump(model, "sign_model.pkl")
