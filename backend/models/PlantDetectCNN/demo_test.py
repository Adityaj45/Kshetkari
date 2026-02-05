#!/usr/bin/env python3
"""
Demo script to test the Plant Detection Model functionality
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
from PIL import Image
import requests
import time
import json

def create_demo_model():
    """Create a demo model for testing"""
    print("Creating demo model...")
    
    img_size = 256
    num_classes = 38
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, 
                            input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        layers.Dense(256, activation='relu'),
        Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    return model

def test_model_inference(model):
    """Test model inference with dummy data"""
    print("\n" + "="*50)
    print("TESTING MODEL INFERENCE")
    print("="*50)
    
    # Create dummy image data
    dummy_image = np.random.rand(1, 256, 256, 3)
    
    # Test single prediction
    print("Testing single image prediction...")
    start_time = time.time()
    prediction = model.predict(dummy_image, verbose=0)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction sum: {np.sum(prediction[0]):.6f}")
    print(f"Top prediction: Class {np.argmax(prediction[0])} ({np.max(prediction[0]):.4f})")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    batch_size = 8
    dummy_batch = np.random.rand(batch_size, 256, 256, 3)
    
    start_time = time.time()
    batch_predictions = model.predict(dummy_batch, verbose=0)
    batch_time = time.time() - start_time
    
    print(f"Batch inference time: {batch_time*1000:.2f} ms")
    print(f"Time per image: {batch_time/batch_size*1000:.2f} ms")
    print(f"Throughput: {batch_size/batch_time:.2f} images/sec")
    
    return True

def test_web_api():
    """Test the web API if it's running"""
    print("\n" + "="*50)
    print("TESTING WEB API")
    print("="*50)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✓ Health check passed")
            print(f"  Status: {health_data['status']}")
            print(f"  Model loaded: {health_data['model_loaded']}")
        else:
            print("✗ Health check failed")
            return False
        
        # Test model info endpoint
        response = requests.get(f"{base_url}/model_info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print("✓ Model info retrieved")
            print(f"  Total parameters: {model_info['total_parameters']:,}")
            print(f"  Number of classes: {model_info['num_classes']}")
            print(f"  Image size: {model_info['image_size']}")
        else:
            print("✗ Model info failed")
        
        print("\n✓ Web API is running and accessible!")
        print(f"  Visit {base_url} to test the web interface")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Web API not accessible: {e}")
        print("  Make sure to run: python run_tests.py --web")
        return False

def run_performance_benchmark():
    """Run a quick performance benchmark"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    model = create_demo_model()
    
    # Warmup
    dummy_image = np.random.rand(1, 256, 256, 3)
    for _ in range(5):
        _ = model.predict(dummy_image, verbose=0)
    
    # Benchmark different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    results = {}
    
    for batch_size in batch_sizes:
        dummy_batch = np.random.rand(batch_size, 256, 256, 3)
        
        times = []
        for _ in range(10):
            start_time = time.time()
            _ = model.predict(dummy_batch, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        time_per_image = avg_time / batch_size
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            'avg_time': avg_time,
            'time_per_image': time_per_image,
            'throughput': throughput
        }
        
        print(f"Batch size {batch_size:2d}: {time_per_image*1000:6.2f} ms/image, "
              f"{throughput:6.2f} images/sec")
    
    return results

def main():
    """Main demo function"""
    print("Plant Detection Model - Demo & Test Suite")
    print("="*60)
    
    # Test model creation and inference
    model = create_demo_model()
    test_model_inference(model)
    
    # Test web API
    test_web_api()
    
    # Run performance benchmark
    benchmark_results = run_performance_benchmark()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run unit tests: python run_tests.py --test")
    print("2. Run benchmarks: python run_tests.py --benchmark")
    print("3. Start web app: python run_tests.py --web")
    print("4. Visit http://localhost:5000 to test the web interface")
    
    return True

if __name__ == "__main__":
    main()