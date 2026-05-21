import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib
import time

# ==========================================
# Professional UI Helper Functions
# ==========================================

def draw_smart_corner_box(img, x, y, w, h, color, thickness=3, length=25):
    """
    Draw professional HUD-style targeting corners around the detected face.
    """

    # Top-left corner
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)

    # Top-right corner
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    # Create a light transparent overlay inside the face region
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        color,
        -1
    )

    # Blend the transparent overlay with the original image
    cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

def draw_glass_panel(img, text, pos,
                     bg_color=(0, 0, 0),
                     text_color=(255, 255, 255)):
    """
    Draw a transparent glass-style panel behind the text
    to improve visual appearance and readability.
    """

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 1

    # Calculate text dimensions to dynamically size the background panel
    (text_w, text_h), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x, y = pos

    # Create overlay for the transparency effect
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x - 10, y - text_h - 10),
        (x + text_w + 10, y + 10),
        bg_color,
        -1
    )

    # Apply transparency to the background rectangle
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Draw the text over the transparent panel
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

# ==========================================
# AI Model Configuration
# ==========================================

# Parameters for Local Binary Pattern (LBP) feature extraction
radius = 2
n_points = 16

print("[INFO] Loading Security Model ")

try:
    # Load the pre-trained machine learning model (SVM)
    model = joblib.load("models/liveness_svm_model.pkl")

    # Load the feature scaler used during training
    scaler = joblib.load("models/liveness_scaler.pkl")

except Exception as e:
    print("[ERROR] Model files are missing. Please ensure they exist in the 'models' directory.")
    exit()

# Load the OpenCV Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

print("[INFO] Initializing Camera System...")

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

# Set the desired camera resolution (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# System State Variables
# ==========================================

# Initial system state
state = "IDLE"

# Timer for the scanning process
scan_start_time = 0

# List to keep track of detected faces across frames
face_trackers = []

# Variable to hold the final processed frame
final_frame = None

print("[INFO] System Online.")

# ==========================================
# Main Program Loop
# ==========================================

while True:

    # --------------------------------------
    # RESULT MODE
    # --------------------------------------
    # If the scan is complete, freeze the screen and show results
    if state == "RESULT":

        cv2.imshow(
            "BIOMETRIC ANTI-SPOOFING SYSTEM",
            final_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Press 'R' to reset and start a new scan
        if key == ord('r') or key == ord('R'):
            state = "IDLE"
            face_trackers = []

        # Press 'Q' to quit the application
        elif key == ord('q') or key == ord('Q'):
            break

        continue

    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Break the loop if the frame cannot be read
    if not ret:
        break

    # Flip the image horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Keep a clean copy of the frame before drawing any UI elements
    clean_frame = frame.copy()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    height, width = frame.shape[:2]

    # ======================================
    # IDLE MODE
    # ======================================
    if state == "IDLE":

        # Draw a detection box for every detected face
        for (x, y, w, h) in faces:

            draw_smart_corner_box(
                frame,
                x,
                y,
                w,
                h,
                (255, 255, 0)
            )

            cv2.putText(
                frame,
                "TARGET LOCKED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        # Display the system's operational status instructions
        draw_glass_panel(
            frame,
            "SYSTEM ARMED - PRESS [S] OR [SPACE] TO INITIATE SCAN",
            (20, height - 30),
            bg_color=(20, 20, 20),
            text_color=(0, 255, 255)
        )

        cv2.imshow(
            "BIOMETRIC ANTI-SPOOFING SYSTEM",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Press 'S' or SPACE to begin the biometric scan
        if key == ord('s') or key == ord('S') or key == 32:

            state = "SCANNING"
            scan_start_time = time.time()
            face_trackers = []

        # Press 'Q' to quit the application
        elif key == ord('q') or key == ord('Q'):
            break

    # ======================================
    # SCANNING MODE
    # ======================================
    elif state == "SCANNING":

        # Calculate how long the scan has been running
        elapsed = time.time() - scan_start_time

        # Calculate remaining time (assuming a 3-second scan)
        remain = max(0, 3.0 - elapsed)

        # Process each detected face during the scan
        for (x, y, w, h) in faces:

            # Find the center coordinates of the detected face
            cx, cy = x + w // 2, y + h // 2

            # Extract the Region of Interest (ROI) containing the face
            face_roi = gray[y:y+h, x:x+w]

            # Standardize the size of the face image for the model
            face_roi_resized = cv2.resize(face_roi, (100, 100))

            # Extract Local Binary Pattern (LBP) texture features
            lbp = local_binary_pattern(
                face_roi_resized,
                n_points,
                radius,
                method="uniform"
            )

            # Build a histogram of the LBP features
            (hist, _) = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, n_points + 3),
                range=(0, n_points + 2)
            )

            hist = hist.astype("float")

            # Normalize the histogram
            hist /= (hist.sum() + 1e-7)

            # Reshape features to match the model's expected input shape
            features = hist.reshape(1, -1)

            # Scale the features using the pre-loaded scaler
            features_scaled = scaler.transform(features)

            # Predict the probability of the face being real (liveness)
            prob_real = model.predict_proba(features_scaled)[0][1]

            # ----------------------------------
            # Face Tracking Logic
            # ----------------------------------

            matched_idx = -1
            min_dist = 100

            # Match the current face to existing tracked faces based on distance
            for i, t in enumerate(face_trackers):

                dist = np.sqrt(
                    (cx - t['cx'])**2 +
                    (cy - t['cy'])**2
                )

                if dist < min_dist:
                    min_dist = dist
                    matched_idx = i

            # If a match is found, update the existing face tracker
            if matched_idx != -1:

                face_trackers[matched_idx]['cx'] = cx
                face_trackers[matched_idx]['cy'] = cy
                face_trackers[matched_idx]['probs'].append(prob_real)
                face_trackers[matched_idx]['last_box'] = (x, y, w, h)

            # If no match is found, create a new tracker instance
            else:

                face_trackers.append({
                    'cx': cx,
                    'cy': cy,
                    'probs': [prob_real],
                    'last_box': (x, y, w, h)
                })

            # ----------------------------------
            # Draw Scanning Visual Effects
            # ----------------------------------

            draw_smart_corner_box(
                frame,
                x,
                y,
                w,
                h,
                (0, 165, 255)
            )

            draw_glass_panel(
                frame,
                "ANALYZING BIOMETRICS...",
                (x, y - 15),
                bg_color=(0, 0, 0),
                text_color=(0, 165, 255)
            )

            # Create an animated laser scanning effect moving up and down
            scan_speed = 2.0

            laser_y = int(
                y +
                (np.sin(elapsed * scan_speed) + 1) / 2 * h
            )

            cv2.line(
                frame,
                (x, laser_y),
                (x + w, laser_y),
                (0, 165, 255),
                2
            )

        # Display the remaining scan time at the bottom of the screen
        draw_glass_panel(
            frame,
            f"PROCESSING... TIME REMAINING: {remain:.1f}s",
            (20, height - 30),
            text_color=(0, 0, 255)
        )

        # ==================================
        # FINAL DECISION
        # ==================================
        # Evaluate the results once the scan duration reaches 3 seconds
        if elapsed >= 3.0:

            # Switch to the clean frame to remove laser/scanning effects
            final_frame = clean_frame.copy()

            for t in face_trackers:

                # Ensure we have enough frames to make a confident decision
                if len(t['probs']) > 5:

                    # Calculate the average liveness probability across all scanned frames
                    avg_prob = np.mean(t['probs'])

                    bx, by, bw, bh = t['last_box']

                    # DECISION: REAL FACE (Access Granted)
                    if avg_prob > 0.5:

                        color = (0, 255, 0)
                        status_text = "ACCESS GRANTED"
                        label = f"REAL LIVENESS: {avg_prob * 100:.1f}%"

                    # DECISION: SPOOF FACE (Access Denied)
                    else:

                        color = (0, 0, 255)
                        status_text = "SECURITY BREACH DETECTED"
                        label = f"SPOOF ATTACK: {(1.0 - avg_prob) * 100:.1f}%"

                    # Draw the final decision box around the face
                    draw_smart_corner_box(
                        final_frame,
                        bx,
                        by,
                        bw,
                        bh,
                        color,
                        thickness=4
                    )

                    # Display the exact probability percentage
                    draw_glass_panel(
                        final_frame,
                        label,
                        (bx, by - 15),
                        bg_color=(0, 0, 0),
                        text_color=color
                    )

                    # Display the final access status text above the bounding box
                    cv2.putText(
                        final_frame,
                        status_text,
                        (bx, by - 45),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        color,
                        1,
                        cv2.LINE_AA
                    )

            # Display restart instructions
            draw_glass_panel(
                final_frame,
                "SCAN COMPLETE - PRESS [R] TO RESTART SYSTEM",
                (20, height - 30),
                bg_color=(25, 25, 25),
                text_color=(255, 255, 255)
            )

            # Transition to the result display state
            state = "RESULT"

        # Display the live scanning feed
        cv2.imshow(
            "BIOMETRIC ANTI-SPOOFING SYSTEM",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Press 'Q' to quit the application during scanning
        if key == ord('q') or key == ord('Q'):
            break

# ==========================================
# Cleanup
# ==========================================

# Release the webcam hardware resources
cap.release()

# Close all active OpenCV windows safely
cv2.destroyAllWindows()