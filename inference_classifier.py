import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# This script runs the real-time application.
# It captures video, detects hands, and uses the trained model to predict the sign.

# 1. LOAD TRAINED MODEL
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Run train_mediapipe.py first.")
    exit()

# 2. SETUP MEDIAPIPE HAND DETECTOR
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    # This should have been downloaded already
    print(f"Error: {model_path} not found.")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
# Detect up to 2 hands to match our new training data
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.3)
detector = vision.HandLandmarker.create_from_options(options)

# 3. OPEN CAMERA
# Try multiple indices to find a working camera
cap = None
for i in range(5):
    temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        print(f"Camera opened at index {i}")
        cap = temp_cap
        break
    temp_cap.release()

if cap is None:
    print("Error: Could not open any camera.")
    exit()

print("Press 'q' to quit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands
    detection_result = detector.detect(mp_image)
    
    # 4. PROCESS DETECTED HANDS
    if detection_result.hand_landmarks:
        # Loop through all detected hands to draw and extract data
        for hand_landmarks in detection_result.hand_landmarks:
            # Visualize (draw dots on hands)
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * W), int(landmark.y * H)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) 
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Extract normalized coordinates
            for landmark in hand_landmarks:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # 5. PREDICT SIGN
        # Our model expects 84 features (2 hands).
        
        # If only 1 hand detected, pad with zeros (42 -> 84)
        if len(data_aux) == 42:
            data_aux.extend([0.0] * 42)
            
        # Only predict if we have valid feature length (84)
        if len(data_aux) == 84:
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
                
                # Draw bounding box around hands
                x1 = int(min(x_) * W) - 40
                y1 = int(min(y_) * H) - 40
                x2 = int(max(x_) * W) + 40
                y2 = int(max(y_) * H) + 40

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, str(predicted_character), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            except Exception as e:
                pass

    # Display the frame
    cv2.imshow('frame', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
