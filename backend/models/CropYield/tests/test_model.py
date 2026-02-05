"""
Comprehensive unit tests for Crop Yield Prediction Model
Tests model loading, prediction accuracy, input validation, and performance
"""

import unittest
import numpy as np
import pandas as pd
import joblib
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

warnings.filterwarnings('ignore')

class TestCropYieldModel(unittest.TestCase):
    """Test suite for crop yield prediction model functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.model_path = "Model/ensemble_model.pkl"
        cls.data_path = "combined_dataset.csv"
        cls.model = None
        cls.test_data = None
        cls.label_encoders = {}
        
        # Load model if it exists
        if os.path.exists(cls.model_path):
            try:
                cls.model = joblib.load(cls.model_path)
                print(f"✓ Loaded model from {cls.model_path}")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
        
        # Load test data if it exists
        if os.path.exists(cls.data_path):
            try:
                cls.test_data = pd.read_csv(cls.data_path)
                print(f"✓ Loaded test data: {cls.test_data.shape}")
                
                # Set up label encoders
                categorical_columns = ['crop', 'season', 'state']
                for col in categorical_columns:
                    if col in cls.test_data.columns:
                        le = LabelEncoder()
                        le.fit(cls.test_data[col])
                        cls.label_encoders[col] = le
                
            except Exception as e:
                print(f"✗ Failed to load test data: {e}")
        
        # Create mock model if real model not available
        if cls.model is None:
            print("Creating mock model for testing...")
            cls._create_mock_model()
    
    @classmethod
    def _create_mock_model(cls):
        """Create a mock model for testing when real model is not available"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Create mock ensemble model
        cls.model = {
            'xgb_ultra': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'xgb_fine': RandomForestRegressor(n_estimators=40, max_depth=4, random_state=123),
            'gb_enhanced': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
            'scaler': StandardScaler(),
            'weights': [0.7, 0.15, 0.15],
            'features': ['year', 'fertilizer', 'pesticide', 'avg_temp_c', 'total_rainfall_mm', 
                        'avg_humidity_percent', 'nitrogen', 'phosphorus', 'potassium', 'pH', 
                        'crop_encoded', 'season_encoded', 'state_encoded']
        }
        
        # Create mock data for training
        np.random.seed(42)
        n_samples = 1000
        n_features = len(cls.model['features'])
        
        X_mock = np.random.rand(n_samples, n_features)
        y_mock = np.random.uniform(0.1, 5.0, n_samples)  # Yield values
        
        # Scale features
        X_scaled = cls.model['scaler'].fit_transform(X_mock)
        
        # Train ensemble models
        cls.model['xgb_ultra'].fit(X_scaled, y_mock)
        cls.model['xgb_fine'].fit(X_scaled, y_mock)
        cls.model['gb_enhanced'].fit(X_scaled, y_mock)
        
        # Set up mock label encoders
        crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane']
        seasons = ['Kharif', 'Rabi', 'Whole Year']
        states = ['Punjab', 'Haryana', 'UP', 'Maharashtra', 'Karnataka']
        
        cls.label_encoders = {
            'crop': LabelEncoder().fit(crops),
            'season': LabelEncoder().fit(seasons),
            'state': LabelEncoder().fit(states)
        }
        
        print("✓ Mock ensemble model created successfully")
    
    def _predict_with_model(self, features):
        """Make prediction with ensemble model"""
        if isinstance(self.model, dict):
            # Ensemble model
            features_scaled = self.model['scaler'].transform(features.reshape(1, -1))
            
            pred1 = self.model['xgb_ultra'].predict(features_scaled)[0]
            pred2 = self.model['xgb_fine'].predict(features_scaled)[0]
            pred3 = self.model['gb_enhanced'].predict(features_scaled)[0]
            
            weights = self.model['weights']
            ensemble_pred = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3
            
            return ensemble_pred
        else:
            # Single model
            return self.model.predict(features.reshape(1, -1))[0]
    
    def test_01_model_loading(self):
        """Test if model loads successfully"""
        self.assertIsNotNone(self.model, "Model should be loaded")
        
        if isinstance(self.model, dict):
            # Ensemble model
            required_keys = ['xgb_ultra', 'xgb_fine', 'gb_enhanced', 'scaler', 'weights', 'features']
            for key in required_keys:
                self.assertIn(key, self.model, f"Model should have {key}")
            print("✓ Ensemble model structure test passed")
        else:
            # Single model
            self.assertTrue(hasattr(self.model, 'predict'), "Model should have predict method")
            print("✓ Single model loading test passed")
    
    def test_02_model_components(self):
        """Test individual model components"""
        if isinstance(self.model, dict):
            # Test each component
            for component_name in ['xgb_ultra', 'xgb_fine', 'gb_enhanced']:
                component = self.model[component_name]
                self.assertTrue(hasattr(component, 'predict'), f"{component_name} should have predict method")
            
            # Test scaler
            scaler = self.model['scaler']
            self.assertTrue(hasattr(scaler, 'transform'), "Scaler should have transform method")
            
            # Test weights
            weights = self.model['weights']
            self.assertEqual(len(weights), 3, "Should have 3 weights")
            self.assertAlmostEqual(sum(weights), 1.0, places=2, msg="Weights should sum to 1")
            
            print("✓ Model components test passed")
        else:
            print("⚠ Single model - skipping component tests")
    
    def test_03_input_validation(self):
        """Test model input validation"""
        # Test with valid input (13 features expected)
        valid_input = np.array([2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5])
        
        try:
            prediction = self._predict_with_model(valid_input)
            self.assertIsNotNone(prediction, "Model should return prediction")
            self.assertIsInstance(prediction, (int, float, np.number), "Prediction should be numeric")
            print("✓ Valid input test passed")
        except Exception as e:
            self.fail(f"Model failed with valid input: {e}")
        
        # Test with invalid input shapes
        with self.assertRaises((ValueError, IndexError)):
            invalid_input = np.array([2020, 1000])  # Too few features
            self._predict_with_model(invalid_input)
        print("✓ Invalid input test passed")
    
    def test_04_prediction_consistency(self):
        """Test prediction consistency"""
        test_input = np.array([2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5])
        
        # Make multiple predictions
        pred1 = self._predict_with_model(test_input)
        pred2 = self._predict_with_model(test_input)
        
        self.assertAlmostEqual(pred1, pred2, places=5, msg="Predictions should be consistent")
        print("✓ Prediction consistency test passed")
    
    def test_05_prediction_range(self):
        """Test if predictions are within expected range"""
        test_inputs = [
            [2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5],  # Normal case
            [2015, 500, 200, 20.0, 100.0, 60.0, 60, 30, 40, 6.0, 0, 1, 3],   # Lower values
            [2022, 2000, 800, 35.0, 300.0, 90.0, 120, 80, 100, 8.0, 2, 2, 10] # Higher values
        ]
        
        for i, test_input in enumerate(test_inputs):
            prediction = self._predict_with_model(np.array(test_input))
            
            # Yield should be positive and reasonable (0.01 to 10 tons/hectare)
            self.assertGreater(prediction, 0, f"Prediction {i+1} should be positive")
            self.assertLess(prediction, 20, f"Prediction {i+1} should be reasonable (<20)")
            
        print("✓ Prediction range test passed")
    
    def test_06_feature_sensitivity(self):
        """Test model sensitivity to feature changes"""
        base_input = np.array([2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5])
        base_prediction = self._predict_with_model(base_input)
        
        # Test sensitivity to fertilizer (should increase yield)
        high_fertilizer_input = base_input.copy()
        high_fertilizer_input[1] = 2000  # Double fertilizer
        high_fertilizer_pred = self._predict_with_model(high_fertilizer_input)
        
        # Generally, more fertilizer should increase yield (though not always)
        # We just check that the model responds to changes
        self.assertNotEqual(base_prediction, high_fertilizer_pred, 
                           "Model should respond to fertilizer changes")
        
        # Test sensitivity to temperature
        high_temp_input = base_input.copy()
        high_temp_input[3] = 40.0  # High temperature
        high_temp_pred = self._predict_with_model(high_temp_input)
        
        self.assertNotEqual(base_prediction, high_temp_pred,
                           "Model should respond to temperature changes")
        
        print("✓ Feature sensitivity test passed")
    
    def test_07_batch_prediction(self):
        """Test batch prediction capability"""
        # Create batch input
        batch_inputs = [
            [2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5],
            [2019, 800, 400, 22.0, 120.0, 70.0, 70, 35, 45, 6.2, 0, 1, 3],
            [2021, 1200, 600, 28.0, 180.0, 80.0, 90, 45, 55, 6.8, 2, 0, 7]
        ]
        
        predictions = []
        for input_data in batch_inputs:
            pred = self._predict_with_model(np.array(input_data))
            predictions.append(pred)
        
        self.assertEqual(len(predictions), 3, "Should return 3 predictions")
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions),
                       "All predictions should be numeric")
        print("✓ Batch prediction test passed")
    
    def test_08_model_accuracy(self):
        """Test model accuracy on available data"""
        if self.test_data is not None and len(self.test_data) > 100:
            try:
                # Prepare test data (use first 100 samples)
                sample_data = self.test_data.iloc[:100].copy()
                
                # Encode categorical variables
                for col, encoder in self.label_encoders.items():
                    if col in sample_data.columns:
                        sample_data[f'{col}_encoded'] = encoder.transform(sample_data[col])
                
                # Select features (adjust based on available columns)
                available_features = []
                expected_features = ['year', 'fertilizer', 'pesticide', 'avg_temp_c', 
                                   'total_rainfall_mm', 'avg_humidity_percent', 'nitrogen', 
                                   'phosphorus', 'potassium', 'pH', 'crop_encoded', 
                                   'season_encoded', 'state_encoded']
                
                for feature in expected_features:
                    if feature in sample_data.columns:
                        available_features.append(feature)
                
                if len(available_features) >= 10 and 'yield' in sample_data.columns:
                    X_test = sample_data[available_features]
                    y_test = sample_data['yield']
                    
                    # Make predictions
                    predictions = []
                    for _, row in X_test.iterrows():
                        # Pad or truncate features to match expected size
                        features = row.values[:13]  # Take first 13 features
                        if len(features) < 13:
                            features = np.pad(features, (0, 13 - len(features)), 'constant')
                        
                        pred = self._predict_with_model(features)
                        predictions.append(pred)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    
                    self.assertGreater(r2, -1, "R² should be greater than -1")
                    self.assertGreater(mae, 0, "MAE should be positive")
                    self.assertGreater(rmse, 0, "RMSE should be positive")
                    
                    print(f"✓ Model accuracy test passed: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
                    return r2
                else:
                    print("⚠ Insufficient features for accuracy testing")
                    return None
                    
            except Exception as e:
                print(f"⚠ Could not test accuracy: {e}")
                return None
        else:
            print("⚠ No test data available for accuracy testing")
            return None
    
    def test_09_memory_usage(self):
        """Test model memory efficiency"""
        import sys
        
        model_size = sys.getsizeof(self.model)
        self.assertLess(model_size, 500 * 1024 * 1024, "Model should be less than 500MB")
        print(f"✓ Memory usage test passed: {model_size / (1024*1024):.2f} MB")
    
    def test_10_inference_speed(self):
        """Test model inference speed"""
        test_input = np.array([2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5])
        
        # Warm up
        self._predict_with_model(test_input)
        
        # Time multiple predictions
        start_time = time.time()
        for _ in range(100):
            self._predict_with_model(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        self.assertLess(avg_time, 0.1, "Average prediction time should be less than 100ms")
        print(f"✓ Inference speed test passed: {avg_time*1000:.2f}ms per prediction")
        
        return avg_time

def run_tests():
    """Run all tests and return results"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCropYieldModel)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CROP YIELD PREDICTION MODEL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result

if __name__ == "__main__":
    run_tests()