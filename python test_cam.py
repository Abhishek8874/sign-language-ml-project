import cv2
import joblib
import numpy as np

model = joblib.load("sign_model.pkl")
label_to_letter = {i: chr(65 + i) for i in range(26)}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened:", cap.isOpened())
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Test", frame)
    key = cv2.waitKey(1)
    
    if key == ord('q') or key == 27:  # Q or ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Done")