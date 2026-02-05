#!/usr/bin/env python3
"""
Simplified benchmark for Plant Detection Model (without dataset dependency)
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
import psutil
import json
from datetime import datetime

class SimpleBenchmark:
    """Simplified benchmarking without dataset dependency"""
    
    def __init__(self):
        self.img_size = 256
        self.batch_size = 32
        self.num_classes = 38
        self.model = None
        self.benchmark_results = {}
        
    def create_model(self):
        """Create model architecture"""
        print("Creating model architecture...")
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(self.img_size, self.img_size, 3))
        base_model.trainable = False
        
        self.model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            layers.Dense(256, activation='relu'),
            Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        print(f"✓ Model created successfully!")
        print(f"  Total parameters: {self.model.count_params():,}")
        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Output shape: {self.model.output_shape}")
        
    def benchmark_inference_speed(self):
        """Benchmark model inference speed"""
        print("\n" + "="*60)
        print("INFERENCE SPEED BENCHMARK")
        print("="*60)
        
        # Single image inference
        dummy_image = np.random.rand(1, self.img_size, self.img_size, 3)
        
        # Warmup
        print("Warming up model...")
        for _ in range(10):
            _ = self.model.predict(dummy_image, verbose=0)
        
        # Single image timing
        print("Testing single image inference...")
        times = []
        for _ in range(50):
            start_time = time.time()
            _ = self.model.predict(dummy_image, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        single_inference_time = np.mean(times)
        single_inference_std = np.std(times)
        
        # Batch inference
        print("Testing batch inference...")
        batch_sizes = [1, 4, 8, 16, 32]
        batch_times = {}
        
        for batch_size in batch_sizes:
            dummy_batch = np.random.rand(batch_size, self.img_size, self.img_size, 3)
            
            # Warmup
            for _ in range(3):
                _ = self.model.predict(dummy_batch, verbose=0)
            
            # Timing
            batch_inference_times = []
            for _ in range(10):
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
        
        print(f"✓ Single image inference: {single_inference_time*1000:.2f} ± {single_inference_std*1000:.2f} ms")
        print(f"✓ Throughput (single): {1.0/single_inference_time:.2f} FPS")
        
        print("\nBatch Performance:")
        for batch_size, metrics in batch_times.items():
            print(f"  Batch size {batch_size:2d}: {metrics['mean_per_image']*1000:6.2f} ms/image, "
                  f"Throughput: {metrics['throughput']:6.2f} images/sec")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n" + "="*60)
        print("MEMORY USAGE BENCHMARK")
        print("="*60)
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
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
            'memory_increase_mb': memory_increase
        }
        
        print(f"✓ Model size: {model_size_mb:.2f} MB")
        print(f"✓ Initial memory: {initial_memory:.2f} MB")
        print(f"✓ Memory during inference: {inference_memory:.2f} MB")
        print(f"✓ Memory increase: {memory_increase:.2f} MB")
    
    def benchmark_robustness(self):
        """Test model robustness with various input conditions"""
        print("\n" + "="*60)
        print("ROBUSTNESS BENCHMARK")
        print("="*60)
        
        # Test with different noise levels
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        robustness_results = {}
        
        dummy_image = np.random.rand(10, self.img_size, self.img_size, 3)
        baseline_pred = self.model.predict(dummy_image, verbose=0)
        
        print("Testing robustness to noise...")
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
        
        print("✓ Robustness to noise:")
        for noise_level, consistency in robustness_results.items():
            print(f"    Noise level {noise_level}: {consistency:.3f} consistency")
    
    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "="*60)
        print("BENCHMARK REPORT")
        print("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'total_parameters': self.model.count_params(),
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'num_classes': self.num_classes
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / 1024**3, 2),
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
            },
            'benchmark_results': self.benchmark_results
        }
        
        # Save report
        report_path = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Benchmark report saved to: {report_path}")
        
        # Print summary
        print("\nSUMMARY:")
        if 'inference_speed' in self.benchmark_results:
            print(f"  Inference Speed: {self.benchmark_results['inference_speed']['single_image_time_ms']:.2f} ms")
            print(f"  Max Throughput: {max([v['throughput'] for v in self.benchmark_results['inference_speed']['batch_performance'].values()]):.2f} images/sec")
        if 'memory_usage' in self.benchmark_results:
            print(f"  Model Size: {self.benchmark_results['memory_usage']['model_size_mb']:.2f} MB")
            print(f"  Memory Usage: {self.benchmark_results['memory_usage']['inference_memory_mb']:.2f} MB")
        
        return report
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Plant Detection Model - Simplified Benchmark Suite")
        print("="*60)
        
        self.create_model()
        self.benchmark_inference_speed()
        self.benchmark_memory_usage()
        self.benchmark_robustness()
        
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return report

if __name__ == "__main__":
    benchmark = SimpleBenchmark()
    benchmark.run_full_benchmark()