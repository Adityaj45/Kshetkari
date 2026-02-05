"""
Comprehensive benchmarking suite for Crop Yield Prediction Model
Tests performance metrics, scalability, and resource usage
"""

import time
import numpy as np
import pandas as pd
import joblib
import os
import psutil
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

class CropYieldBenchmark:
    """Benchmark suite for crop yield prediction model"""
    
    def __init__(self):
        self.model_path = "Model/ensemble_model.pkl"
        self.data_path = "combined_dataset.csv"
        self.model = None
        self.test_data = None
        self.label_encoders = {}
        self.results = {}
        
        self._load_model_and_data()
    
    def _load_model_and_data(self):
        """Load model and test data"""
        # Load model
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"✓ Loaded model from {self.model_path}")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                self._create_mock_model()
        else:
            print("Model file not found, creating mock model...")
            self._create_mock_model()
        
        # Load test data
        if os.path.exists(self.data_path):
            try:
                self.test_data = pd.read_csv(self.data_path)
                
                # Set up label encoders
                categorical_columns = ['crop', 'season', 'state']
                for col in categorical_columns:
                    if col in self.test_data.columns:
                        le = LabelEncoder()
                        le.fit(self.test_data[col])
                        self.label_encoders[col] = le
                
                print(f"✓ Loaded test data: {self.test_data.shape}")
            except Exception as e:
                print(f"✗ Failed to load test data: {e}")
                self._create_mock_data()
        else:
            print("Test data not found, creating mock data...")
            self._create_mock_data()
    
    def _create_mock_model(self):
        """Create mock ensemble model for testing"""
        print("Creating mock ensemble model...")
        
        self.model = {
            'xgb_ultra': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'xgb_fine': RandomForestRegressor(n_estimators=40, max_depth=4, random_state=123),
            'gb_enhanced': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
            'scaler': StandardScaler(),
            'weights': [0.7, 0.15, 0.15],
            'features': ['year', 'fertilizer', 'pesticide', 'avg_temp_c', 'total_rainfall_mm', 
                        'avg_humidity_percent', 'nitrogen', 'phosphorus', 'potassium', 'pH', 
                        'crop_encoded', 'season_encoded', 'state_encoded']
        }
        
        # Train on mock data
        np.random.seed(42)
        X_mock = np.random.rand(1000, 13)
        y_mock = np.random.uniform(0.1, 5.0, 1000)
        
        X_scaled = self.model['scaler'].fit_transform(X_mock)
        self.model['xgb_ultra'].fit(X_scaled, y_mock)
        self.model['xgb_fine'].fit(X_scaled, y_mock)
        self.model['gb_enhanced'].fit(X_scaled, y_mock)
        
        print("✓ Mock ensemble model created")
    
    def _create_mock_data(self):
        """Create mock test data"""
        np.random.seed(42)
        n_samples = 5000
        
        crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'soybean', 'barley']
        seasons = ['Kharif', 'Rabi', 'Whole Year']
        states = ['Punjab', 'Haryana', 'UP', 'Maharashtra', 'Karnataka', 'Gujarat', 'MP']
        
        self.test_data = pd.DataFrame({
            'crop': np.random.choice(crops, n_samples),
            'year': np.random.randint(1997, 2023, n_samples),
            'season': np.random.choice(seasons, n_samples),
            'state': np.random.choice(states, n_samples),
            'area': np.random.uniform(1000, 100000, n_samples),
            'production': np.random.uniform(500, 200000, n_samples),
            'fertilizer': np.random.uniform(100, 3000, n_samples),
            'pesticide': np.random.uniform(50, 1000, n_samples),
            'yield': np.random.uniform(0.1, 8.0, n_samples),
            'avg_temp_c': np.random.uniform(15, 40, n_samples),
            'total_rainfall_mm': np.random.uniform(50, 400, n_samples),
            'avg_humidity_percent': np.random.uniform(40, 95, n_samples),
            'nitrogen': np.random.randint(20, 150, n_samples),
            'phosphorus': np.random.randint(10, 100, n_samples),
            'potassium': np.random.randint(15, 120, n_samples),
            'pH': np.random.uniform(4.5, 9.0, n_samples)
        })
        
        # Set up label encoders
        for col in ['crop', 'season', 'state']:
            le = LabelEncoder()
            le.fit(self.test_data[col])
            self.label_encoders[col] = le
        
        print("✓ Mock test data created")
    
    def _predict_with_model(self, features_array):
        """Make prediction with ensemble model"""
        if isinstance(self.model, dict):
            # Ensemble model
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            
            features_scaled = self.model['scaler'].transform(features_array)
            
            pred1 = self.model['xgb_ultra'].predict(features_scaled)
            pred2 = self.model['xgb_fine'].predict(features_scaled)
            pred3 = self.model['gb_enhanced'].predict(features_scaled)
            
            weights = self.model['weights']
            ensemble_pred = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3
            
            return ensemble_pred
        else:
            # Single model
            return self.model.predict(features_array)
    
    def benchmark_inference_speed(self):
        """Benchmark model inference speed"""
        print("\n" + "="*50)
        print("INFERENCE SPEED BENCHMARK")
        print("="*50)
        
        test_samples = [1, 10, 100, 1000]
        speed_results = {}
        
        for n_samples in test_samples:
            # Create test data
            test_features = np.random.rand(n_samples, 13)
            
            # Warm up
            self._predict_with_model(test_features[:1])
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                predictions = self._predict_with_model(test_features)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = n_samples / avg_time
            
            speed_results[n_samples] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'throughput_samples_per_sec': throughput,
                'time_per_sample_ms': (avg_time / n_samples) * 1000
            }
            
            print(f"Batch size {n_samples:4d}: {avg_time*1000:6.2f}ms ± {std_time*1000:5.2f}ms "
                  f"({throughput:6.1f} samples/sec, {(avg_time/n_samples)*1000:.3f}ms per sample)")
        
        self.results['inference_speed'] = speed_results
        return speed_results
    
    def benchmark_accuracy(self):
        """Benchmark model accuracy"""
        print("\n" + "="*50)
        print("ACCURACY BENCHMARK")
        print("="*50)
        
        # Prepare test data
        sample_data = self.test_data.iloc[:2000].copy()
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in sample_data.columns:
                sample_data[f'{col}_encoded'] = encoder.transform(sample_data[col])
        
        # Select features
        feature_columns = ['year', 'fertilizer', 'pesticide', 'avg_temp_c', 'total_rainfall_mm', 
                          'avg_humidity_percent', 'nitrogen', 'phosphorus', 'potassium', 'pH', 
                          'crop_encoded', 'season_encoded', 'state_encoded']
        
        available_features = [col for col in feature_columns if col in sample_data.columns]
        
        if len(available_features) >= 10 and 'yield' in sample_data.columns:
            sample_sizes = [100, 500, 1000, min(2000, len(sample_data))]
            accuracy_results = {}
            
            for n_samples in sample_sizes:
                X_test = sample_data[available_features].iloc[:n_samples]
                y_test = sample_data['yield'].iloc[:n_samples]
                
                # Pad features to 13 if needed
                if X_test.shape[1] < 13:
                    padding = np.zeros((X_test.shape[0], 13 - X_test.shape[1]))
                    X_test_padded = np.hstack([X_test.values, padding])
                else:
                    X_test_padded = X_test.values[:, :13]
                
                # Make predictions
                start_time = time.time()
                predictions = self._predict_with_model(X_test_padded)
                prediction_time = time.time() - start_time
                
                # Calculate metrics
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 0.001))) * 100
                
                accuracy_results[n_samples] = {
                    'r2_score': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'prediction_time_ms': prediction_time * 1000
                }
                
                print(f"Sample size {n_samples:4d}: R²={r2:.4f}, MAE={mae:.4f}, "
                      f"RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
            self.results['accuracy'] = accuracy_results
            return accuracy_results
        else:
            print("⚠ Insufficient data for accuracy benchmark")
            return {}
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n" + "="*50)
        print("MEMORY USAGE BENCHMARK")
        print("="*50)
        
        import sys
        
        # Model size
        model_size_bytes = sys.getsizeof(self.model)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Memory usage during prediction
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Memory during large batch prediction
        large_batch = np.random.rand(5000, 13)
        predictions = self._predict_with_model(large_batch)
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        memory_results = {
            'model_size_mb': model_size_mb,
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - baseline_memory
        }
        
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Baseline memory: {baseline_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Memory increase: {peak_memory - baseline_memory:.2f} MB")
        
        self.results['memory_usage'] = memory_results
        return memory_results
    
    def benchmark_robustness(self):
        """Test model robustness with edge cases"""
        print("\n" + "="*50)
        print("ROBUSTNESS BENCHMARK")
        print("="*50)
        
        robustness_results = {}
        
        # Test with extreme values
        extreme_cases = [
            [1997, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # All zeros (except year)
            [2022, 5000, 2000, 50, 500, 100, 200, 150, 200, 14, 6, 2, 10],  # All high values
            [2010, 1000, 500, 25, 150, 75, 80, 40, 50, 6.5, 1, 0, 5],  # Normal case
        ]
        
        case_names = ['extreme_low', 'extreme_high', 'normal']
        
        for i, case in enumerate(extreme_cases):
            try:
                prediction = self._predict_with_model(np.array(case))
                robustness_results[case_names[i]] = {
                    'success': True,
                    'prediction': float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction),
                    'error': None
                }
                print(f"✓ {case_names[i]}: Prediction = {prediction}")
            except Exception as e:
                robustness_results[case_names[i]] = {
                    'success': False,
                    'prediction': None,
                    'error': str(e)
                }
                print(f"✗ {case_names[i]}: Error = {e}")
        
        # Test with noisy data
        clean_data = np.random.rand(100, 13)
        clean_data[:, 0] = np.random.randint(1997, 2023, 100)  # Realistic years
        
        noise_levels = [0.1, 0.2, 0.5]
        for noise_level in noise_levels:
            noisy_data = clean_data + np.random.normal(0, noise_level * np.std(clean_data, axis=0), clean_data.shape)
            # Keep years realistic
            noisy_data[:, 0] = np.clip(noisy_data[:, 0], 1990, 2030)
            
            try:
                predictions = self._predict_with_model(noisy_data)
                robustness_results[f'noise_{noise_level}'] = {
                    'success': True,
                    'num_predictions': len(predictions),
                    'avg_prediction': float(np.mean(predictions)),
                    'std_prediction': float(np.std(predictions)),
                    'error': None
                }
                print(f"✓ Noise level {noise_level}: {len(predictions)} predictions, "
                      f"avg={np.mean(predictions):.3f}, std={np.std(predictions):.3f}")
            except Exception as e:
                robustness_results[f'noise_{noise_level}'] = {
                    'success': False,
                    'num_predictions': 0,
                    'error': str(e)
                }
                print(f"✗ Noise level {noise_level}: Error = {e}")
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def benchmark_feature_sensitivity(self):
        """Test sensitivity to individual features"""
        print("\n" + "="*50)
        print("FEATURE SENSITIVITY BENCHMARK")
        print("="*50)
        
        feature_names = ['year', 'fertilizer', 'pesticide', 'avg_temp_c', 'total_rainfall_mm', 
                        'avg_humidity_percent', 'nitrogen', 'phosphorus', 'potassium', 'pH', 
                        'crop_encoded', 'season_encoded', 'state_encoded']
        
        base_input = np.array([2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5])
        base_prediction = self._predict_with_model(base_input)[0]
        
        sensitivity_results = {}
        
        for i, feature in enumerate(feature_names):
            # Test with ±20% change (or ±1 for categorical features)
            if i >= 10:  # Categorical features
                changes = [-1, 1]
            else:
                changes = [-0.2, -0.1, 0.1, 0.2]
            
            feature_sensitivity = {}
            
            for change in changes:
                modified_input = base_input.copy().astype(float)
                
                if i >= 10:  # Categorical features
                    modified_input[i] = max(0, modified_input[i] + change)
                else:
                    modified_input[i] *= (1 + change)
                
                try:
                    new_prediction = self._predict_with_model(modified_input)[0]
                    prediction_change = abs(new_prediction - base_prediction)
                    relative_change = prediction_change / max(abs(base_prediction), 0.001)
                    
                    feature_sensitivity[f'{change:+.1f}'] = {
                        'prediction': float(new_prediction),
                        'absolute_change': float(prediction_change),
                        'relative_change': float(relative_change)
                    }
                except Exception as e:
                    feature_sensitivity[f'{change:+.1f}'] = {
                        'prediction': None,
                        'absolute_change': None,
                        'relative_change': None,
                        'error': str(e)
                    }
            
            sensitivity_results[feature] = feature_sensitivity
            
            # Calculate average sensitivity
            valid_changes = [v for v in feature_sensitivity.values() 
                           if isinstance(v, dict) and v.get('relative_change') is not None]
            if valid_changes:
                avg_sensitivity = np.mean([v['relative_change'] for v in valid_changes])
                print(f"{feature:20s}: Avg sensitivity = {avg_sensitivity:.4f}")
            else:
                print(f"{feature:20s}: No valid sensitivity measurements")
        
        self.results['feature_sensitivity'] = sensitivity_results
        return sensitivity_results
    
    def run_all_benchmarks(self):
        """Run all benchmark tests"""
        print("CROP YIELD PREDICTION MODEL BENCHMARK SUITE")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all benchmarks
        self.benchmark_inference_speed()
        self.benchmark_accuracy()
        self.benchmark_memory_usage()
        self.benchmark_robustness()
        self.benchmark_feature_sensitivity()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Speed summary
        if 'inference_speed' in self.results:
            speed_1 = self.results['inference_speed'].get(1, {})
            speed_1000 = self.results['inference_speed'].get(1000, {})
            print(f"Single prediction: {speed_1.get('avg_time_ms', 0):.2f}ms")
            print(f"Batch throughput: {speed_1000.get('throughput_samples_per_sec', 0):.1f} samples/sec")
        
        # Accuracy summary
        if 'accuracy' in self.results:
            best_r2 = max(acc['r2_score'] for acc in self.results['accuracy'].values() if 'r2_score' in acc)
            print(f"Best R² score: {best_r2:.4f} ({best_r2*100:.2f}%)")
        
        # Memory summary
        if 'memory_usage' in self.results:
            mem = self.results['memory_usage']
            print(f"Model size: {mem.get('model_size_mb', 0):.2f} MB")
            print(f"Memory usage: {mem.get('memory_increase_mb', 0):.2f} MB increase")
        
        # Robustness summary
        if 'robustness' in self.results:
            robust_tests = self.results['robustness']
            success_count = sum(1 for test in robust_tests.values() 
                              if isinstance(test, dict) and test.get('success', False))
            total_tests = len(robust_tests)
            print(f"Robustness: {success_count}/{total_tests} tests passed")
        
        print(f"\nOverall Status: {'✓ PASSED' if self._overall_status() else '✗ ISSUES DETECTED'}")
    
    def _overall_status(self):
        """Determine overall benchmark status"""
        # Check if critical benchmarks passed
        if 'accuracy' in self.results:
            best_r2 = max(acc.get('r2_score', -999) for acc in self.results['accuracy'].values())
            if best_r2 < 0:  # Negative R² indicates poor performance
                return False
        
        if 'robustness' in self.results:
            robust_tests = self.results['robustness']
            success_count = sum(1 for test in robust_tests.values() 
                              if isinstance(test, dict) and test.get('success', False))
            if success_count < len(robust_tests) * 0.7:  # Less than 70% robustness tests passed
                return False
        
        return True
    
    def save_results(self, filename=None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'crop_yield_benchmark_{timestamp}.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        clean_results = clean_dict(self.results)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
        return filename

def main():
    """Main function to run benchmarks"""
    benchmark = CropYieldBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.save_results()
    return results

if __name__ == "__main__":
    main()