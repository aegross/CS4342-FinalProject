import kagglehub
import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

# Configuration
DATASET_DIR = "animals10/raw-img"  # The original dataset directory
OUTPUT_DIR = "cleaned_animals_dataset"
IMAGE_SIZE = (48, 48)  # Standardized image size
IMAGE_FORMAT = "jpeg"  # Desired image format

def rename_folders(input_dir):
    """Renames category folders based on the translation dictionary."""
    translate = {
        "cane": "dog", "cavallo": "horse", "elefante": "elephant",
        "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
        "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
        "ragno": "spider"
    }
    
    renamed_folders = {}

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            new_category_name = translate.get(category, category)
            new_category_path = os.path.join(input_dir, new_category_name)
            if new_category_name != category and not os.path.exists(new_category_path):
                os.rename(category_path, new_category_path)
                renamed_folders[category] = new_category_name

    return renamed_folders  # Return the mapping for later use

def clean_and_standardize_images(input_dir, output_dir, image_size, image_format):
    """Cleans and standardizes images by resizing, converting to grayscale, and saving in a specified format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure image_format is lowercase for consistency
    image_format = image_format.lower()

    for category in tqdm(os.listdir(input_dir), desc="Cleaning images"):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if os.path.isdir(category_path):
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            image_count = 0
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)

                try:
                    with Image.open(img_path) as img:
                        # Convert to grayscale (1 channel)
                        img = img.convert("RGB").resize(image_size)  # "L" mode = 8-bit grayscale

                        # Save as specified format
                        output_img_path = os.path.join(output_category_path, f"{os.path.splitext(img_name)[0]}.{image_format}")
                        img.save(output_img_path, image_format.upper())
                        image_count += 1

                except UnidentifiedImageError:
                    print(f"Skipping {img_name} in {category}: Cannot identify image format")
                except Exception as e:
                    print(f"Skipping {img_name} in {category}: {e}")

            if image_count == 0:
                print(f"Warning: No images processed in category '{category}'!")

def split_dataset(input_dir, train_dir, test_dir, test_size=0.2):
    """Splits the dataset into training and testing sets."""
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for category in tqdm(os.listdir(input_dir), desc="Splitting dataset"):
        category_path = os.path.join(input_dir, category)
        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)

        if os.path.isdir(category_path):
            if not os.path.exists(train_category_path):
                os.makedirs(train_category_path)
            if not os.path.exists(test_category_path):
                os.makedirs(test_category_path)

            images = os.listdir(category_path)
            if len(images) == 0:
                print(f"Warning: Skipping {category}, no images found.")
                continue

            train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
            for img_name in train_images:
                shutil.copy(os.path.join(category_path, img_name), os.path.join(train_category_path, img_name))
            for img_name in test_images:
                shutil.copy(os.path.join(category_path, img_name), os.path.join(test_category_path, img_name))

def save_as_npy(data_dir, output_file):
    """Converts image dataset into NumPy array format and ensures data consistency."""
    images = []
    labels = []
    categories = sorted(os.listdir(data_dir))  # Ensure consistent order for labels
    label_map = {category: idx for idx, category in enumerate(categories)}

    for category in tqdm(categories, desc=f"Converting {output_file}"):
        category_path = os.path.join(data_dir, category)

        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB").resize(IMAGE_SIZE)
                        img_array = np.array(img) / 255.0  # Normalize pixel values to [0,1]
                        images.append(img_array)
                        labels.append(label_map[category])  # Ensure label is added correctly
                except Exception as e:
                    print(f"Skipping {img_name} in {category}: {e}")

    # Convert lists to numpy arrays
    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)

    # Debugging print to check consistency
    print(f"Saving {output_file}: {images_np.shape[0]} images, {labels_np.shape[0]} labels")

    if images_np.shape[0] != labels_np.shape[0]:
        raise ValueError(f"Mismatch: {images_np.shape[0]} images but {labels_np.shape[0]} labels!")

    np.save(output_file + "_images.npy", images_np)
    np.save(output_file + "_labels.npy", labels_np)

if __name__ == "__main__":
    # Get the current working directory
    current_directory = os.getcwd()
    dataset_name = "animals10"
    new_path = os.path.join(current_directory, dataset_name)

    if not os.path.exists(new_path):
        path = kagglehub.dataset_download("alessiocorrado99/animals10")
        if os.path.exists(path):
            shutil.move(path, new_path)
            print(f"Dataset moved to: {new_path}")
        else:
            print("Dataset download failed or path not found.")

    # Always rename folders before processing
    print("Renaming folders to English names...")
    rename_folders(DATASET_DIR)

    print("Starting dataset cleaning and preparation...")
    CLEANED_DIR = os.path.join(OUTPUT_DIR, "standardized")
    clean_and_standardize_images(DATASET_DIR, CLEANED_DIR, IMAGE_SIZE, IMAGE_FORMAT)

    TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
    TEST_DIR = os.path.join(OUTPUT_DIR, "test")
    split_dataset(CLEANED_DIR, TRAIN_DIR, TEST_DIR)

    print("Dataset preparation complete!")
    print(f"Training dataset stored in: {TRAIN_DIR}")
    print(f"Testing dataset stored in: {TEST_DIR}")

    # Convert to .npy files
    print("Converting datasets to .npy format...")
    save_as_npy(TRAIN_DIR, os.path.join(OUTPUT_DIR, "train_data"))
    save_as_npy(TEST_DIR, os.path.join(OUTPUT_DIR, "test_data"))

    print("Dataset preparation complete!")
    print(f"Training dataset: {OUTPUT_DIR}/train_data_images.npy & train_data_labels.npy")
    print(f"Testing dataset: {OUTPUT_DIR}/test_data_images.npy & test_data_labels.npy")
