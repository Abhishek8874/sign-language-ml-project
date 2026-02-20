import cv2
import os

output_dir = "my_dataset"
os.makedirs(output_dir, exist_ok=True)

letters = list('ABCDEFGHIKLMNOPQRSTUVWXY')
cap = cv2.VideoCapture(0)

for letter in letters:
    folder = os.path.join(output_dir, letter)
    os.makedirs(folder, exist_ok=True)
    
    print(f"\nGet ready to show letter: {letter}")
    print("Press SPACE to start capturing 20 images...")
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        x, y, bw, bh = w//2-150, h//2-150, 300, 300
        cv2.rectangle(frame, (x,y), (x+bw, y+bh), (0,255,0), 2)
        cv2.putText(frame, f"Show letter: {letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.putText(frame, "Press SPACE to capture", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Collect Data", frame)
        
        if cv2.waitKey(30) & 0xFF == ord(' '):
            break

    count = 0
    while count < 20:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        x, y, bw, bh = w//2-150, h//2-150, 300, 300
        
        hand = frame[y:y+bh, x:x+bw]
        img_path = os.path.join(folder, f"{count}.png")
        cv2.imwrite(img_path, hand)
        count += 1
        
        cv2.rectangle(frame, (x,y), (x+bw, y+bh), (0,255,0), 2)
        cv2.putText(frame, f"{letter}: {count}/20", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        cv2.imshow("Collect Data", frame)
        cv2.waitKey(100)
    
    print(f"Captured 20 images for {letter}")

cap.release()
cv2.destroyAllWindows()
print("Done! Now update train.py to use my_dataset folder")