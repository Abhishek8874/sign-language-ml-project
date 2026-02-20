import os
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# This script processes the raw images to extract hand landmarks.
# It supports both 1-handed and 2-handed signs by standardizing the feature vector size.

# 1. SETUP MEDIAPIPE HAND DETECTOR
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Please download it.")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
# Configure to detect up to 2 hands (common in ISL)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.3)
detector = vision.HandLandmarker.create_from_options(options)

# 2. LOCATE DATASET
DATA_DIR = './isl_dataset/unzipped'
if not os.path.exists(DATA_DIR):
     DATA_DIR = './isl_dataset/data'

print(f"Looking for data in: {DATA_DIR}")

data = []
labels = []

if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' not found.")
    exit()

# 3. LOOP THROUGH EACH CLASS (A, B, C...)
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing class: {dir_}")
    
    count = 0 
    # Process up to 80 images per class for speed (User requested "fast")
    for img_path in os.listdir(dir_path):
        if count >= 80:
            break
            
        data_aux = []
        x_ = []
        y_ = []

        full_path = os.path.join(dir_path, img_path)
        
        # Load image for MediaPipe
        try:
             mp_image = mp.Image.create_from_file(full_path)
        except Exception as e:
             # Skip corrupt images
             continue

        # Detect hands
        detection_result = detector.detect(mp_image)
        
        # If hands are found, extract coordinates
        if detection_result.hand_landmarks:
            # We want a consistent feature vector of size 84 (2 hands * 21 landmarks * 2 coords)
            # Strategy: Always produce 84 features.
            # If 1 hand found: fill first 42, pad rest with 0.
            # If 2 hands found: fill all 84.
            
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks:
                    # Normalize relative to min x, y to be position invariant
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
            
            # PADDING A SINGLE HAND TO MATCH 2-HAND SIZE
            if len(data_aux) == 42:
                data_aux.extend([0.0] * 42)
            
            # Only save if we have exactly 84 features (valid data)
            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(dir_)
                count += 1

# 4. SAVE PROCESSED DATA
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"Done! Processed {len(data)} images. Data saved to 'data.pickle'.")
