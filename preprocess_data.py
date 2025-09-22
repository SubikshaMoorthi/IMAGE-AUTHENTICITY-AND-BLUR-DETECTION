import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import subprocess
import zipfile

# Define the Kaggle dataset path (replace with your dataset path)
# The format is: <your-kaggle-username>/<your-dataset-slug>
KAGGLE_DATASET = "subiksham27/image-authenticity-and-blur-detection-dataset" 

# Define local directories
DATASET_DIR = "dataset"
UPLOADED_ZIP_DIR = "IMAGE-AUTHENTICITY-AND-BLUR-DETECTION" # Name of the folder after unzipping

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """Downloads the dataset from Kaggle if it doesn't exist."""
    if not os.path.exists(DATASET_DIR):
        print("Dataset directory not found. Downloading from Kaggle...")
        try:
            # Check if Kaggle API is configured
            if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
                raise FileNotFoundError("Kaggle API key not found. Please follow the setup steps.")
            
            # Use subprocess to run the Kaggle CLI command
            # The '--unzip' flag extracts the files automatically
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "--unzip"], check=True)
            
            # The Kaggle download command typically unzips into a folder with the dataset name.
            # You might need to adjust this depending on how your dataset is structured.
            # For this example, we assume it unzips into a folder named 'IMAGE-AUTHENTICITY-AND-BLUR-DETECTION'
            if os.path.exists(UPLOADED_ZIP_DIR):
                os.rename(UPLOADED_ZIP_DIR, DATASET_DIR)
            else:
                # If the dataset's top-level directory is different, you'll need to rename it
                print(f"Dataset extracted to a different directory. Please rename it to '{DATASET_DIR}'.")

        except FileNotFoundError:
            print("Kaggle CLI not found. Please install it with 'pip install kaggle'.")
            exit()
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset from Kaggle: {e}")
            print("Please ensure your dataset name is correct and it is publicly accessible or you have permissions.")
            exit()
    else:
        print("Dataset already exists. Skipping download.")

# Run the download function before setting up generators
load_and_preprocess_data()

# Set parameters for generators
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators for Authenticity
authenticity_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

auth_train_generator = authenticity_datagen.flow_from_directory(
    f'{DATASET_DIR}/Authenticity',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

auth_val_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    f'{DATASET_DIR}/Authenticity',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Data generators for Detection
detection_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

det_train_generator = detection_datagen.flow_from_directory(
    f'{DATASET_DIR}/Detection',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

det_val_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    f'{DATASET_DIR}/Detection',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("Data generators created successfully.")