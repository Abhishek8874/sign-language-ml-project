
import cv2

def test_camera():
    print("Testing cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # CAP_DSHOW is often faster/better on Windows
        if cap.isOpened():
            print(f"Camera found at index {i}")
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working. Resolution: {frame.shape}")
                cv2.imshow(f'Camera {i}', frame)
                cv2.waitKey(1000)
                cap.release()
            else:
                print(f"Camera {i} opened but failed to read frame.")
        else:
            print(f"No camera at index {i}")
    
    cv2.destroyAllWindows()
    print("Test complete.")

if __name__ == "__main__":
    test_camera()
