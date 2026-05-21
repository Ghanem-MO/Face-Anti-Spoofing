import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# ==========================================================
# LBP CONFIGURATION
# ==========================================================

# Radius around each pixel for LBP analysis.
# Larger radius = larger texture area analyzed.
radius = 2

# Number of neighboring points around the center pixel.
# Usually calculated as:
# 8 * radius
# Here:
# 8 * 2 = 16
n_points = 16


# ==========================================================
# LOAD FACE DETECTION MODEL
# ==========================================================

# Haar Cascade is a pre-trained face detector included in OpenCV.
# It detects frontal human faces inside images.
# The XML file contains thousands of trained face patterns.

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# ==========================================================
# FEATURE EXTRACTION FUNCTION
# ==========================================================

def extract_lbp_features(image_path):

    
    # Read image from disk
    image = cv2.imread(image_path)

    # corrupted image
    if image is None:
        return None

    # ------------------------------------------------------
    # Convert image to grayscale
    # ------------------------------------------------------
    # LBP does not need color information.
    # It only analyzes texture and intensity patterns.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------
    # Detect faces in image
    # ------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,

        # Scale factor:
        # The image is resized multiple times during detection.
        # 1.1 means reduce image size by 10% each step.
        scaleFactor=1.1,

        # Minimum number of detections required
        # before considering a region as a face.
        # Higher value = fewer false detections.
        minNeighbors=5
    )

    # ------------------------------------------------------
    # Skip images without detected faces
    # ------------------------------------------------------
    # This prevents training on:
    # - background
    # - walls
    # - random objects
    if len(faces) == 0:
        return None

    # ------------------------------------------------------
    # Select the largest detected face
    # ------------------------------------------------------
    # Sometimes multiple faces appear.
    # We assume the biggest face is the main subject.
    (x, y, w, h) = max(
        faces,

        # Calculate face area:
        # width * height
        key=lambda rect: rect[2] * rect[3]
    )

    # ------------------------------------------------------
    # Extract face region only
    # ------------------------------------------------------
    # ROI = Region Of Interest
    # Crop only the face area.
    face_roi = gray[y:y+h, x:x+w]

    # ------------------------------------------------------
    # Resize face image
    # ------------------------------------------------------
    # All images must have the same size
    # before feature extraction.
    face_roi = cv2.resize(face_roi, (100, 100))

    # ======================================================
    # EXTRACT LBP TEXTURE FEATURES
    # ======================================================

    lbp = local_binary_pattern(
        face_roi,

        # Number of neighboring points
        n_points,

        # Radius around each pixel
        radius,

        # Uniform LBP:
        # Faster and more stable than basic LBP
        method="uniform"
    )

    # ------------------------------------------------------
    # Build histogram from LBP image
    # ------------------------------------------------------
    # Histogram converts texture patterns into numerical features.
    (hist, _) = np.histogram(

        # Flatten image into 1D array
        lbp.ravel(),

        # Histogram bins
        bins=np.arange(0, n_points + 3),

        # Value range
        range=(0, n_points + 2)
    )

    # Convert histogram to float
    hist = hist.astype("float")

    # ------------------------------------------------------
    # Normalize histogram
    # ------------------------------------------------------
    # This makes all feature vectors comparable
    # regardless of image brightness or size.
    hist /= (hist.sum() + 1e-7)

    # Return extracted features
    return hist


# ==========================================================
# LOAD DATASET
# ==========================================================

print("[INFO] Loading dataset and extracting features from detected faces only...")

# Feature vectors will be stored here
data = []

# Labels will be stored here
labels = []

# ----------------------------------------------------------
# Dataset labels
# ----------------------------------------------------------
# Real face  -> 1
# Fake face  -> 0
dataset_paths = {
    "real": 1,
    "spoof": 0
}

# Root dataset directory
base_dir = "my_dataset"

# ==========================================================
# PROCESS DATASET
# ==========================================================

# Loop through:
# - real folder
# - spoof folder
for folder_name, label in dataset_paths.items():

    # Create full folder path
    folder_path = os.path.join(base_dir, folder_name)

    # Skip missing folders
    if not os.path.exists(folder_path):
        continue

    # Get all image filenames
    files = os.listdir(folder_path)

    print(f"\n[INFO] Processing {folder_name} images...")

    # ------------------------------------------------------
    # Process all images with progress bar
    # ------------------------------------------------------
    for file_name in tqdm(files):

        # Full image path
        img_path = os.path.join(folder_path, file_name)

        # Extract LBP features
        features = extract_lbp_features(img_path)

        # Store only valid extracted features
        if features is not None:

            # Save feature vector
            data.append(features)

            # Save image label
            labels.append(label)

# ==========================================================
# CONVERT TO NUMPY ARRAYS
# ==========================================================

data = np.array(data)
labels = np.array(labels)

print(f"\n[INFO] Successfully extracted faces from {len(data)} images.")


# ==========================================================
# SPLIT DATASET
# ==========================================================

print("[INFO] Splitting data into training and testing sets...")

(trainX, testX, trainY, testY) = train_test_split(

    # Features
    data,

    # Labels
    labels,

    # 20% for testing
    test_size=0.2,

    # Fixed random seed
    # Ensures reproducible results
    random_state=42
)


# ==========================================================
# FEATURE SCALING
# ==========================================================

print("[INFO] Scaling features for better SVM performance...")

# Create scaler object
scaler = StandardScaler()

# ----------------------------------------------------------
# Learn scaling parameters from training data
# then transform training data
# ----------------------------------------------------------
trainX = scaler.fit_transform(trainX)

# ----------------------------------------------------------
# Apply same scaling to testing data
# ----------------------------------------------------------
testX = scaler.transform(testX)


# ==========================================================
# TRAIN SVM MODEL
# ==========================================================

print("[INFO] Training Advanced SVM Model (RBF Kernel)...")

# Create SVM classifier
model = SVC(

    # RBF kernel:
    # Excellent for non-linear data
    kernel="rbf",

    # Regularization parameter
    # Larger value = stronger fitting
    C=10.0,

    # Automatically scale gamma
    gamma="scale",

    # Enable probability predictions
    probability=True
)

# ----------------------------------------------------------
# Train the AI model
# ----------------------------------------------------------
model.fit(trainX, trainY)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

print("[INFO] Evaluating model performance...")

# Predict labels for testing data
predictions = model.predict(testX)

# Calculate accuracy score
acc = accuracy_score(testY, predictions)

print(f"\n[SUCCESS] Model Accuracy on Test Data: {acc * 100:.2f}%")


# ==========================================================
# SAVE TRAINED MODEL
# ==========================================================

# Save trained SVM model
joblib.dump(model, "models/liveness_svm_model.pkl")

# Save scaler object
# Important because future data must use same scaling
joblib.dump(scaler, "models/liveness_scaler.pkl")

print("[INFO] Model and scaler saved successfully!")