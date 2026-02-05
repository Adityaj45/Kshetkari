import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
import json
from datetime import datetime
import psutil
import GPUtil

class ModelBenchmark:
    """Comprehensive benchmarking suite for Plant Detection Model"""
    
    def __init__(self, model_path=None, data_path="../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"):
        self.model_path = model_path
        self.data_path = data_path
        self.img_size = 256
        self.batch_size = 32
        self.num_classes = 38
        self.model = None
        self.test_generator = None
        self.class_names = []
        self.benchmark_results = {}
        
    def load_model(self):
        """Load the trained model"""
        if self.model_path and os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            # Create model architecture for testing
            self.model = self.create_model_architecture()
        
        print(f"Model loaded successfully. Total parameters: {self.model.count_params():,}")
        
    def create_model_architecture(self):
        """Create model architecture for benchmarking"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(self.img_size, self.img_size, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            layers.Dense(256, activation='relu'),
            Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def setup_test_data(self):
        """Setup test data generator"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Use validation split for testing
        if os.path.exists(os.path.join(self.data_path, "valid")):
            test_dir = os.path.join(self.data_path, "valid")
        else:
            test_dir = os.path.join(self.data_path, "train")
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(self.test_generator.class_indices.keys())
        print(f"Test data loaded: {self.test_generator.samples} samples, {len(self.class_names)} classes")
    
    def benchmark_inference_speed(self, num_samples=1000):
        """Benchmark model inference speed"""
        print("\n=== INFERENCE SPEED BENCHMARK ===")
        
        # Single image inference
        dummy_image = np.random.rand(1, self.img_size, self.img_size, 3)
        
        # Warmup
        for _ in range(10):
            _ = self.model.predict(dummy_image, verbose=0)
        
        # Single image timing
        times = []
        for _ in range(100):
            start_time = time.time()
            _ = self.model.predict(dummy_image, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        single_inference_time = np.mean(times)
        single_inference_std = np.std(times)
        
        # Batch inference
        batch_sizes = [1, 8, 16, 32, 64]
        batch_times = {}
        
        for batch_size in batch_sizes:
            dummy_batch = np.random.rand(batch_size, self.img_size, self.img_size, 3)
            
            # Warmup
            for _ in range(5):
                _ = self.model.predict(dummy_batch, verbose=0)
            
            # Timing
            batch_inference_times = []
            for _ in range(20):
                start_time = time.time()
                _ = self.model.predict(dummy_batch, verbose=0)
                end_time = time.time()
                batch_inference_times.append((end_time - start_time) / batch_size)
            
            batch_times[batch_size] = {
                'mean_per_image': np.mean(batch_inference_times),
                'std_per_image': np.std(batch_inference_times),
                'throughput': batch_size / np.mean([t * batch_size for t in batch_inference_times])
            }
        
        self.benchmark_results['inference_speed'] = {
            'single_image_time_ms': single_inference_time * 1000,
            'single_image_std_ms': single_inference_std * 1000,
            'batch_performance': batch_times,
            'fps_single': 1.0 / single_inference_time
        }
        
        print(f"Single image inference: {single_inference_time*1000:.2f} ± {single_inference_std*1000:.2f} ms")
        print(f"Throughput (single): {1.0/single_inference_time:.2f} FPS")
        
        for batch_size, metrics in batch_times.items():
            print(f"Batch size {batch_size}: {metrics['mean_per_image']*1000:.2f} ms/image, "
                  f"Throughput: {metrics['throughput']:.2f} images/sec")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n=== MEMORY USAGE BENCHMARK ===")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory if available
        gpu_memory = None
        if tf.config.list_physical_devices('GPU'):
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = gpus[0].memoryUsed
            except:
                pass
        
        # Model memory footprint
        model_size_mb = self.model.count_params() * 4 / 1024 / 1024  # Assuming float32
        
        # Memory during inference
        dummy_batch = np.random.rand(32, self.img_size, self.img_size, 3)
        _ = self.model.predict(dummy_batch, verbose=0)
        
        inference_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = inference_memory - initial_memory
        
        self.benchmark_results['memory_usage'] = {
            'model_size_mb': model_size_mb,
            'initial_memory_mb': initial_memory,
            'inference_memory_mb': inference_memory,
            'memory_increase_mb': memory_increase,
            'gpu_memory_mb': gpu_memory
        }
        
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Memory during inference: {inference_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        if gpu_memory:
            print(f"GPU memory used: {gpu_memory:.2f} MB")
    
    def benchmark_accuracy(self, max_samples=None):
        """Benchmark model accuracy on test data"""
        print("\n=== ACCURACY BENCHMARK ===")
        
        if self.test_generator is None:
            print("Test data not available for accuracy benchmark")
            return
        
        # Limit samples for faster benchmarking
        if max_samples:
            steps = min(max_samples // self.batch_size, self.test_generator.samples // self.batch_size)
        else:
            steps = self.test_generator.samples // self.batch_size
        
        print(f"Evaluating on {steps * self.batch_size} samples...")
        
        # Get predictions
        start_time = time.time()
        predictions = self.model.predict(self.test_generator, steps=steps, verbose=1)
        prediction_time = time.time() - start_time
        
        # Get true labels
        self.test_generator.reset()
        y_true = []
        for i in range(steps):
            _, labels = next(self.test_generator)
            y_true.extend(np.argmax(labels, axis=1))
        
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, 
                                           target_names=self.class_names[:len(set(y_true))],
                                           output_dict=True)
        
        self.benchmark_results['accuracy'] = {
            'overall_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': prediction_time,
            'samples_evaluated': len(y_true),
            'class_report': class_report
        }
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Evaluation time: {prediction_time:.2f} seconds")
        
        return y_true, y_pred
    
    def benchmark_robustness(self):
        """Test model robustness with various input conditions"""
        print("\n=== ROBUSTNESS BENCHMARK ===")
        
        # Test with different noise levels
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        robustness_results = {}
        
        dummy_image = np.random.rand(10, self.img_size, self.img_size, 3)
        baseline_pred = self.model.predict(dummy_image, verbose=0)
        
        for noise_level in noise_levels:
            noisy_image = dummy_image + np.random.normal(0, noise_level, dummy_image.shape)
            noisy_image = np.clip(noisy_image, 0, 1)
            
            noisy_pred = self.model.predict(noisy_image, verbose=0)
            
            # Calculate prediction consistency
            consistency = np.mean([
                np.argmax(baseline_pred[i]) == np.argmax(noisy_pred[i]) 
                for i in range(len(baseline_pred))
            ])
            
            robustness_results[noise_level] = consistency
        
        self.benchmark_results['robustness'] = robustness_results
        
        print("Robustness to noise:")
        for noise_level, consistency in robustness_results.items():
            print(f"  Noise level {noise_level}: {consistency:.3f} consistency")
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        print("\n=== BENCHMARK REPORT ===")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'total_parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'num_classes': self.num_classes
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
            },
            'benchmark_results': self.benchmark_results
        }
        
        # Save report
        report_path = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Benchmark report saved to: {report_path}")
        return report
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting comprehensive model benchmark...")
        
        self.load_model()
        self.setup_test_data()
        
        self.benchmark_inference_speed()
        self.benchmark_memory_usage()
        self.benchmark_accuracy(max_samples=1000)  # Limit for faster benchmarking
        self.benchmark_robustness()
        
        report = self.generate_benchmark_report()
        
        print("\n=== BENCHMARK SUMMARY ===")
        if 'inference_speed' in self.benchmark_results:
            print(f"Inference Speed: {self.benchmark_results['inference_speed']['single_image_time_ms']:.2f} ms")
        if 'memory_usage' in self.benchmark_results:
            print(f"Model Size: {self.benchmark_results['memory_usage']['model_size_mb']:.2f} MB")
        if 'accuracy' in self.benchmark_results:
            print(f"Accuracy: {self.benchmark_results['accuracy']['overall_accuracy']:.4f}")
        
        return report

if __name__ == "__main__":
    # Run benchmark
    benchmark = ModelBenchmark()
    benchmark.run_full_benchmark()