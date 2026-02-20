# import cv2
# import joblib
# import numpy as np

# model = joblib.load("sign_model.pkl")
# label_to_letter = {i: chr(65 + i) for i in range(26)}

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (28, 28))
#     img = img.flatten().reshape(1, -1)
    
#     prediction = model.predict(img)[0]
#     letter = label_to_letter.get(prediction, prediction)
    
#     cv2.putText(frame, f"Sign: {letter}", (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow("Sign Language Detector", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import joblib
import numpy as np

model = joblib.load("sign_model.pkl")
label_to_letter = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
    '6': 'G', '7': 'H', '8': 'I', '9': 'K', '10': 'L', '11': 'M',
    '12': 'N', '13': 'O', '14': 'P', '15': 'Q', '16': 'R', '17': 'S',
    '18': 'T', '19': 'U', '20': 'V', '21': 'W', '22': 'X', '23': 'Y'
}
sign_hints = {
    'A': 'Fist with thumb on side',
    'B': 'Flat hand fingers up',
    'C': 'Curved hand like C',
    'D': 'Index up others curved',
    'E': 'Fingers bent down',
    'F': 'Index and thumb touch',
    'G': 'Index points sideways',
    'H': 'Two fingers sideways',
    'I': 'Pinky up',
    'K': 'Index and middle up',
    'L': 'L shape with hand',
    'M': 'Three fingers over thumb',
    'N': 'Two fingers over thumb',
    'O': 'Fingers form O shape',
    'P': 'Index points down',
    'Q': 'Index and thumb down',
    'R': 'Crossed fingers',
    'S': 'Fist with thumb over',
    'T': 'Thumb between fingers',
    'U': 'Two fingers up together',
    'V': 'Two fingers up spread',
    'W': 'Three fingers up',
    'X': 'Index finger hooked',
    'Y': 'Pinky and thumb out',
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror so it feels natural
    height, width = frame.shape[:2]

    # Centered large box like the training images
    box_size = 300
    x = width // 2 - box_size // 2
    y = height // 2 - box_size // 2
    w = box_size
    h = box_size

    # Draw green box with corner markers
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Corner markers to guide hand placement
    corner_len = 20
    color = (0, 255, 0)
    cv2.line(frame, (x, y), (x+corner_len, y), color, 4)
    cv2.line(frame, (x, y), (x, y+corner_len), color, 4)
    cv2.line(frame, (x+w, y), (x+w-corner_len, y), color, 4)
    cv2.line(frame, (x+w, y), (x+w, y+corner_len), color, 4)
    cv2.line(frame, (x, y+h), (x+corner_len, y+h), color, 4)
    cv2.line(frame, (x, y+h), (x, y+h-corner_len), color, 4)
    cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), color, 4)
    cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), color, 4)

    cv2.putText(frame, "Place hand inside box", (x, y-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Crop and predict
    hand = frame[y:y+h, x:x+w]
    img = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.flatten().reshape(1, -1)

    # prediction = model.predict(img)[0]
    prediction = model.predict(img)[0]
    letter = label_to_letter.get(str(prediction), str(prediction))  # str() added here
    # letter = label_to_letter.get(prediction, str(prediction))
    hint = sign_hints.get(letter, '')

    # Big letter display
    cv2.putText(frame, f"Sign: {letter}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)

    # Hint in yellow
    cv2.putText(frame, f"Tip: {hint}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, "Press Q to quit", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
