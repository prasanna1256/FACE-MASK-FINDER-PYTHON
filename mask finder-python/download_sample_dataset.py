import os
import urllib.request
import zipfile
import shutil
import ssl

# Disable SSL verification for sample downloads (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

def download_and_extract_dataset(url, download_path, extract_path):
    """
    Download and extract a dataset from a URL
    Args:
        url: URL to download from
        download_path: Path to save the downloaded file
        extract_path: Path to extract the contents to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    # Download the file
    print(f"Downloading dataset from {url}")
    try:
        urllib.request.urlretrieve(url, download_path)
        print(f"Downloaded to {download_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

    # Extract the file
    try:
        print(f"Extracting dataset to {extract_path}")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

    return True

def organize_sample_dataset(extracted_path, target_path):
    """
    Organize a sample dataset into with_mask and without_mask folders
    Args:
        extracted_path: Path where dataset was extracted
        target_path: Path to organize data to
    """
    # Create target directories
    with_mask_dir = os.path.join(target_path, 'with_mask')
    without_mask_dir = os.path.join(target_path, 'without_mask')

    os.makedirs(with_mask_dir, exist_ok=True)
    os.makedirs(without_mask_dir, exist_ok=True)

    # Example organization for a common face mask dataset
    # This will need to be adapted based on the exact dataset structure
    try:
        # Check if we have dataset-specific folders
        if os.path.exists(os.path.join(extracted_path, 'with_mask')):
            # Dataset is already organized correctly
            for img in os.listdir(os.path.join(extracted_path, 'with_mask')):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(
                        os.path.join(extracted_path, 'with_mask', img),
                        os.path.join(with_mask_dir, img)
                    )

            for img in os.listdir(os.path.join(extracted_path, 'without_mask')):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(
                        os.path.join(extracted_path, 'without_mask', img),
                        os.path.join(without_mask_dir, img)
                    )
        else:
            # If dataset is organized differently, you'll need to adjust this logic
            print("Dataset structure not recognized. Please organize manually.")
            print(f"Images should be placed in {with_mask_dir} and {without_mask_dir}")
            return False

        print(f"Dataset organized into {with_mask_dir} and {without_mask_dir}")
        return True

    except Exception as e:
        print(f"Error organizing dataset: {e}")
        return False

def main():
    # Sample dataset URL - replace with a real face mask dataset URL
    # For example, you might use: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
    dataset_url = "https://github.com/prajnasb/observations/raw/master/experiements/data.zip"

    # Paths
    download_path = "./downloads/face_mask_dataset.zip"
    extract_path = "./downloads/extracted"
    target_path = "./data"

    # Download and extract
    if download_and_extract_dataset(dataset_url, download_path, extract_path):
        # Organize
        organize_sample_dataset(extract_path, target_path)

        # Cleanup
        print("Cleaning up temporary files...")
        if os.path.exists(download_path):
            os.remove(download_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)

        print("Dataset preparation complete!")
        print(f"Data is ready in {target_path}")
        print("You can now run train_model.py to train the model.")
    else:
        print("Failed to prepare dataset.")
        print("You may need to download and organize the dataset manually.")
        print("Please place images in ./data/with_mask/ and ./data/without_mask/ folders.")

if __name__ == "__main__":
    main()
