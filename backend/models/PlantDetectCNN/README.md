# Plant Disease Detection System

A comprehensive AI-powered plant disease detection system using Convolutional Neural Networks (CNN) with MobileNetV2 architecture. The system achieves 93%+ accuracy in detecting 38 different plant diseases.

## 🌟 Features

- **High Accuracy**: 93%+ accuracy on validation data
- **38 Disease Classes**: Detects diseases across multiple plant species
- **Fast Inference**: Optimized for real-time predictions
- **Web Interface**: User-friendly web application for testing
- **Batch Processing**: Support for multiple image uploads
- **Comprehensive Testing**: Unit tests and performance benchmarks
- **Mobile-Optimized**: Uses MobileNetV2 for efficient deployment

## 🏗️ Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 256x256x3 RGB images
- **Custom Head**: GlobalAveragePooling2D → BatchNormalization → Dense(256) → Dropout(0.5) → Dense(38)
- **Framework**: TensorFlow/Keras 2.20.0
- **Total Parameters**: ~2.2M parameters

## 📁 Project Structure

```
Plant detection (CNN)/
├── Model/
│   ├── model1.ipynb          # Main training notebook
│   └── model1.pkl            # Saved model weights
├── tests/
│   ├── test_model.py         # Unit tests
│   └── benchmark_model.py    # Performance benchmarks
├── web_app/
│   ├── app.py               # Flask web application
│   └── templates/
│       └── index.html       # Web interface
├── archive/                 # Dataset directory
├── requirements.txt         # Python dependencies
├── run_tests.py            # Test runner script
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "Plant detection (CNN)"

# Install dependencies
python run_tests.py --install
```

### 2. Run Tests

```bash
# Run unit tests
python run_tests.py --test

# Run performance benchmarks
python run_tests.py --benchmark

# Run all tests
python run_tests.py --all
```

### 3. Start Web Application

```bash
# Start the web server
python run_tests.py --web
```

The web application will be available at `http://localhost:5000`

## 🧪 Testing Framework

### Unit Tests (`tests/test_model.py`)

Comprehensive unit tests covering:
- Model architecture validation
- Input/output shape verification
- Prediction functionality
- Image preprocessing pipeline
- Data generator configuration
- Layer-by-layer testing
- Batch prediction testing

```bash
# Run specific unit tests
python -m pytest tests/test_model.py -v
```

### Performance Benchmarks (`tests/benchmark_model.py`)

Detailed performance analysis including:
- **Inference Speed**: Single image and batch processing times
- **Memory Usage**: Model size and runtime memory consumption
- **Accuracy Metrics**: Precision, recall, F1-score on test data
- **Robustness Testing**: Performance under various noise conditions
- **System Resource Monitoring**: CPU and GPU utilization

```bash
# Run benchmarks
python tests/benchmark_model.py
```

## 🌐 Web Interface

### Features
- **Single Image Upload**: Drag-and-drop or click to upload
- **Batch Processing**: Upload up to 10 images simultaneously
- **Real-time Results**: Instant predictions with confidence scores
- **Model Information**: View model architecture and supported classes
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Single image prediction
- `POST /batch_upload` - Batch image processing
- `GET /model_info` - Model architecture information
- `GET /health` - Health check endpoint

### Usage Examples

#### Single Image Upload
```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Predictions:', data.predictions);
});
```

#### Batch Upload
```javascript
const formData = new FormData();
files.forEach(file => formData.append('files', file));

fetch('/batch_upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Batch results:', data.results);
});
```

## 📊 Supported Plant Diseases

The model can detect 38 different plant diseases across multiple species:

### Apple Diseases
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy Apple

### Tomato Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy Tomato

### Other Crops
- Corn (Common Rust, Northern Leaf Blight, Cercospora Leaf Spot)
- Grape (Black Rot, Esca, Leaf Blight)
- Potato (Early Blight, Late Blight)
- Pepper (Bacterial Spot)
- And many more...

## 🔧 Configuration

### Model Parameters
```python
IMG_SIZE = 256          # Input image size
BATCH_SIZE = 32         # Training batch size
NUM_CLASSES = 38        # Number of disease classes
DROPOUT_RATE = 0.5      # Dropout rate in dense layer
DENSE_UNITS = 256       # Units in dense layer
```

### Data Augmentation
```python
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
```

## 📈 Performance Metrics

### Current Performance
- **Validation Accuracy**: 93.32%
- **Model Size**: ~2.2M parameters
- **Inference Time**: ~50ms per image (CPU)
- **Memory Usage**: ~100MB runtime

### Benchmark Results
```
Inference Speed: 45.23 ± 5.12 ms
Model Size: 8.7 MB
Accuracy: 93.32%
Precision: 93.15%
Recall: 93.08%
F1-Score: 93.11%
```

## 🛠️ Development

### Adding New Tests
1. Create test functions in `tests/test_model.py`
2. Follow the naming convention `test_*`
3. Use assertions to validate expected behavior
4. Run tests with `python run_tests.py --test`

### Extending the Web Interface
1. Add new routes in `web_app/app.py`
2. Update the HTML template in `web_app/templates/index.html`
3. Test the interface locally
4. Update API documentation

### Model Improvements
1. Experiment with different architectures
2. Adjust hyperparameters
3. Add more data augmentation techniques
4. Implement transfer learning fine-tuning

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Memory Errors**
   - Reduce batch size
   - Use mixed precision training
   - Clear GPU memory between runs

3. **Web App Not Starting**
   - Check port 5000 availability
   - Verify Flask installation
   - Check file permissions

### Error Logs
- Unit test logs: Check pytest output
- Benchmark logs: Saved as JSON files with timestamps
- Web app logs: Check Flask console output

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review test outputs for debugging information

## 🔄 Updates

### Version History
- **v1.0**: Initial release with 93%+ accuracy
- **v1.1**: Added comprehensive testing framework
- **v1.2**: Web interface and batch processing
- **v1.3**: Performance benchmarking suite

---

**Built with ❤️ using TensorFlow, Flask, and modern web technologies**