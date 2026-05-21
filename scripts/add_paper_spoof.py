import cv2
import os
import time

# Make sure the spoof folder exists (we will not delete anything)
os.makedirs('dataset/spoof', exist_ok=True)

# Number of paper spoof images to add (3000 is an excellent amount)
target_count = 3000

cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n[INFO] Safe Paper Spoof Collector Ready.")
print(f"[INFO] This will ADD {target_count} images to your existing dataset without deleting anything.")
print("Instructions:")
print(" - Hold the PRINTED PHOTO in front of the camera.")
print(" - Press 'P' to start capturing PAPER SPOOFS.")
print(" - Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally like a mirror
    frame = cv2.flip(frame, 1)

    cv2.imshow('Add Paper Spoof', frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture printed paper spoof images
    if key == ord('p') or key == ord('P'):
        print("\n[INFO] Starting PAPER SPOOF capture in 2 seconds...")
        print("💡 TIP: Slightly bend the paper, move it closer/farther, and let the room light reflect on it!")
        time.sleep(2)

        for i in range(target_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for saving and displaying
            frame_flipped = cv2.flip(frame, 1)

            # Save image with a unique name so phone spoof images are not overwritten
            cv2.imwrite(
                f'my_dataset/spoof/paper_spoof_{int(time.time()*1000)}.jpg',
                frame_flipped
            )

            # Show progress and red border on screen
            display_frame = frame_flipped.copy()

            cv2.rectangle(
                display_frame,
                (0, 0),
                (display_frame.shape[1], display_frame.shape[0]),
                (0, 0, 255),
                5
            )

            cv2.putText(
                display_frame,
                f"Capturing PAPER: {i+1}/{target_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow('Add Paper Spoof', display_frame)
            cv2.waitKey(1)

        print("\n[SUCCESS] Paper spoof data safely ADDED to dataset/spoof/!")
        break  # Exit after finishing

    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()