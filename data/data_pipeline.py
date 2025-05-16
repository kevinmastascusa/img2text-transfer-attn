import sys
import os

# Add the project directory to the system path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from data.preprocess import preprocess_data

def data_ingestion_pipeline(train_path, captions_path):
    """
    Pipeline for data ingestion and preprocessing.

    Args:
        train_path (str): Path to the training images.
        captions_path (str): Path to the captions file.

    Returns:
        tuple: Preprocessed images and tokenized captions.
    """
    print("Starting data ingestion pipeline...")

    # Step 1: Load and preprocess data
    images, captions = preprocess_data(train_path, captions_path)
    print(f"Loaded {len(images)} images and {len(captions)} captions.")

    # Step 2: Validate data
    for image in images:
        assert image.shape == (3, 224, 224), "Image shape is incorrect."
    for caption in captions:
        assert isinstance(caption, list), "Caption is not tokenized."
        assert len(caption) > 0, "Caption is empty."

    print("Data ingestion pipeline completed successfully.")
    return images, captions

if __name__ == "__main__":
    # Example usage
    train_path = "data/coco/train2017"
    captions_path = "data/coco/annotations/captions.json"
    data_ingestion_pipeline(train_path, captions_path)