import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("sign_model.pkl")

# Load test image
img = cv2.imread("amer_sign2.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28,28))
img = img.flatten()
img = img.reshape(1, -1)

prediction = model.predict(img)


# Convert number to letter
label_to_letter = {i: chr(65 + i) for i in range(26)}
predicted_letter = label_to_letter.get(prediction[0], prediction[0])

print("Predicted Sign:", prediction[0])
