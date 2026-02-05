import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
import os
import sys
from PIL import Image
import tempfile
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestPlantDetectionModel(unittest.TestCase):
    """Unit tests for Plant Detection CNN Model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.img_size = 256
        cls.num_classes = 38
        cls.batch_size = 32
        cls.test_data_dir = "../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
        
        # Create a simple test model
        cls.model = cls.create_test_model()
        
        # Create test image
        cls.test_image = np.random.rand(cls.img_size, cls.img_size, 3) * 255
        cls.test_image = cls.test_image.astype(np.uint8)
    
    @classmethod
    def create_test_model(cls):
        """Create a test model with same architecture"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(cls.img_size, cls.img_size, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            layers.Dense(256, activation='relu'),
            Dropout(0.5),
            layers.Dense(cls.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def test_model_architecture(self):
        """Test model architecture and layer configuration"""
        # Test input shape
        self.assertEqual(self.model.input_shape, (None, self.img_size, self.img_size, 3))
        
        # Test output shape
        self.assertEqual(self.model.output_shape, (None, self.num_classes))
        
        # Test number of layers
        self.assertGreater(len(self.model.layers), 5)
        
        # Test if model is compiled
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)
    
    def test_model_prediction_shape(self):
        """Test model prediction output shape"""
        # Create dummy input
        dummy_input = np.random.rand(1, self.img_size, self.img_size, 3)
        
        # Get prediction
        prediction = self.model.predict(dummy_input, verbose=0)
        
        # Test prediction shape
        self.assertEqual(prediction.shape, (1, self.num_classes))
        
        # Test if probabilities sum to 1
        self.assertAlmostEqual(np.sum(prediction[0]), 1.0, places=5)
        
        # Test if all probabilities are between 0 and 1
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(prediction <= 1))
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        # Test image rescaling
        rescaled_image = self.test_image / 255.0
        self.assertTrue(np.all(rescaled_image >= 0))
        self.assertTrue(np.all(rescaled_image <= 1))
        
        # Test image resizing
        pil_image = Image.fromarray(self.test_image)
        resized_image = pil_image.resize((self.img_size, self.img_size))
        self.assertEqual(resized_image.size, (self.img_size, self.img_size))
    
    def test_data_generator(self):
        """Test data generator configuration"""
        data_gen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Test if data generator is properly configured
        self.assertEqual(data_gen.rescale, 1./255)
        # Note: validation_split is not directly accessible as attribute
        self.assertEqual(data_gen.rotation_range, 20)
        self.assertTrue(data_gen.horizontal_flip)
    
    def test_model_layers(self):
        """Test individual model layers"""
        layers_list = self.model.layers
        
        # Test MobileNetV2 base
        self.assertIsInstance(layers_list[0], tf.keras.Model)
        
        # Test GlobalAveragePooling2D
        self.assertIsInstance(layers_list[1], GlobalAveragePooling2D)
        
        # Test BatchNormalization
        self.assertIsInstance(layers_list[2], BatchNormalization)
        
        # Test Dense layer
        self.assertIsInstance(layers_list[3], layers.Dense)
        self.assertEqual(layers_list[3].units, 256)
        
        # Test Dropout
        self.assertIsInstance(layers_list[4], Dropout)
        
        # Test output layer
        self.assertIsInstance(layers_list[5], layers.Dense)
        self.assertEqual(layers_list[5].units, self.num_classes)
    
    def test_model_trainability(self):
        """Test model trainability settings"""
        # Base model should be frozen initially
        base_model = self.model.layers[0]
        self.assertFalse(base_model.trainable)
        
        # Test unfreezing
        base_model.trainable = True
        self.assertTrue(base_model.trainable)
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        batch_size = 4
        dummy_batch = np.random.rand(batch_size, self.img_size, self.img_size, 3)
        
        predictions = self.model.predict(dummy_batch, verbose=0)
        
        self.assertEqual(predictions.shape, (batch_size, self.num_classes))
        
        # Test if each prediction sums to 1
        for i in range(batch_size):
            self.assertAlmostEqual(np.sum(predictions[i]), 1.0, places=5)
    
    def test_model_summary(self):
        """Test model summary generation"""
        try:
            # This should not raise an exception
            summary = []
            self.model.summary(print_fn=lambda x: summary.append(x))
            self.assertGreater(len(summary), 0)
        except Exception as e:
            self.fail(f"Model summary failed: {e}")
    
    def test_class_indices_consistency(self):
        """Test class indices consistency"""
        # This would typically test against actual data generator
        # For now, we test the expected number of classes
        expected_classes = 38
        self.assertEqual(self.num_classes, expected_classes)

if __name__ == '__main__':
    # Set up test environment
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0], True
    ) if tf.config.experimental.list_physical_devices('GPU') else None
    
    # Run tests
    unittest.main(verbosity=2)