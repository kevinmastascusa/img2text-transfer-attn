import os
import unittest
import tensorflow_datasets as tfds
from data.data_pipeline import preprocess_data

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        # No need for paths as we are using TensorFlow Datasets
        pass

    def test_dataset_loading(self):
        # Test if the dataset is loaded correctly
        images, captions = preprocess_data()
        self.assertGreater(len(images), 0, "No images loaded.")
        self.assertGreater(len(captions), 0, "No captions loaded.")

    def test_image_preprocessing(self):
        # Test if images are preprocessed correctly
        images, _ = preprocess_data()
        for image in images:
            self.assertEqual(image.shape, (3, 224, 224), "Image shape is incorrect.")

    def test_caption_tokenization(self):
        # Test if captions are tokenized correctly
        _, captions = preprocess_data()
        for caption in captions:
            self.assertIsInstance(caption, str, "Caption is not a string.")
            self.assertGreater(len(caption), 0, "Caption is empty.")

if __name__ == "__main__":
    unittest.main()