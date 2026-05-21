import cv2
import os
import shutil
import time

# 1. Delete old dataset completely to clean the project
dataset_dir = "my_dataset"
# if os.path.exists(dataset_dir):
    # print("[INFO] Deleting old dataset completely...")
    # shutil.rmtree(dataset_dir)

# 2. Create new empty dataset folders
os.makedirs('my_dataset/real', exist_ok=True)
os.makedirs('my_dataset/spoof', exist_ok=True)

target_count = 5000  # Huge number of images for each class

cap = cv2.VideoCapture(0)  # Use laptop webcam

print(f"\n[INFO] Ready to build a massive dataset: {target_count} images per class.")
print("Instructions:")
print(" - Press 'R' to start capturing REAL faces.")
print(" - Press 'S' to start capturing SPOOF faces (phone screen).")
print(" - Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera failed.")
        break

    cv2.imshow('Massive Data Collector', frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture 5000 images for real faces
    if key == ord('r') or key == ord('R'):
        print("\n[INFO] Starting REAL capture in 2 seconds...")
        print("💡 TIP: Move your head, change expressions, change distance from camera!")
        time.sleep(2)  # Time to prepare

        for i in range(target_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Save image
            cv2.imwrite(f'my_dataset/real/real_{i}.jpg', frame)

            # Show progress and green border on screen
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 5)
            cv2.putText(display_frame, f"Capturing REAL: {i+1}/{target_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Massive Data Collector', display_frame)
            cv2.waitKey(1)  # Capture speed

        print("\n[SUCCESS] REAL data collection complete!")

    # Capture 5000 spoof (phone screen) images
    elif key == ord('s') or key == ord('S'):
        print("\n[INFO] Starting SPOOF capture in 2 seconds...")
        print("💡 TIP: Move the phone slowly, change angles to catch screen reflections!")
        time.sleep(2)

        for i in range(target_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Save image
            cv2.imwrite(f'my_dataset/spoof/spoof_{i}.jpg', frame)

            # Show progress and red border on screen
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 5)
            cv2.putText(display_frame, f"Capturing SPOOF: {i+1}/{target_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Massive Data Collector', display_frame)
            cv2.waitKey(1)

        print("\n[SUCCESS] SPOOF data collection complete!")

    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()