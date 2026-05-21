import os
import zipfile
import shutil
import importlib

# ----------------------------------------
# Configure Kaggle authentication
# ----------------------------------------
print("[INFO] Authenticating with Kaggle...")

# Store Kaggle API token in environment variable
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_55c6a1b75c2ea2d29fbb4a3af3854336'

# Dynamically import Kaggle library
# This prevents some IDEs from automatically moving the import
kaggle = importlib.import_module('kaggle')

# ----------------------------------------
# Dataset configuration
# ----------------------------------------

# Kaggle dataset name
dataset_name = "maihongtng/face-anti-spoofing"

# Downloaded zip filename
zip_name = "face-anti-spoofing.zip"

# Temporary extraction folder
temp_dir = "temp_dataset"

print("[INFO] Downloading Liveness Dataset from Kaggle...")
print("[INFO] This may take a few minutes.")

try:
    # Download dataset files without automatic extraction
    kaggle.api.dataset_download_files(
        dataset_name,
        path='.',
        unzip=False
    )

    print("[SUCCESS] Download completed.")

except Exception as e:

    # Display download error
    print(f"[ERROR] Download failed: {e}")
    exit()

# ----------------------------------------
# Extract downloaded zip file
# ----------------------------------------
print("[INFO] Extracting files...")

with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# ----------------------------------------
# Function to search for folders recursively
# ----------------------------------------
def find_folder(start_path, target_name):

    for root, dirs, files in os.walk(start_path):

        if target_name in dirs:
            return os.path.join(root, target_name)

    return None

# Locate REAL and SPOOF folders inside extracted dataset
source_real = find_folder(temp_dir, "live")
source_spoof = find_folder(temp_dir, "not_live")

# Destination dataset folders
dest_real = os.path.join("Kaggle_dataset", "real")
dest_spoof = os.path.join("Kaggle_dataset", "spoof")

# ----------------------------------------
# Remove old dataset if it exists
# ----------------------------------------
if os.path.exists("Kaggle_dataset"):
    shutil.rmtree("Kaggle_dataset")

# Create clean dataset folders
os.makedirs(dest_real, exist_ok=True)
os.makedirs(dest_spoof, exist_ok=True)

# ----------------------------------------
# Function to copy dataset images
# ----------------------------------------
def move_files(src, dst):

    # Check if source folder exists
    if src is None:
        print(f"[WARNING] Source folder not found for {dst}!")
        return

    # Get all files in source folder
    files = os.listdir(src)

    print(f"Moving {len(files)} files to {dst}...")

    for file_name in files:

        # Full file path
        full_file_name = os.path.join(src, file_name)

        # Copy only valid files
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst)

# ----------------------------------------
# Organize REAL images
# ----------------------------------------
print("\n[INFO] Organizing REAL face images...")
move_files(source_real, dest_real)

# ----------------------------------------
# Organize SPOOF images
# ----------------------------------------
print("\n[INFO] Organizing SPOOF/FAKE face images...")
move_files(source_spoof, dest_spoof)

# ----------------------------------------
# Clean temporary files
# ----------------------------------------
print("\n[INFO] Cleaning up temporary files...")

# Remove downloaded zip file
if os.path.exists(zip_name):
    os.remove(zip_name)

# Remove extracted temporary folder
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

# ----------------------------------------
# Finish process
# ----------------------------------------
print("\n[SUCCESS] New Liveness Dataset is ready for training!")