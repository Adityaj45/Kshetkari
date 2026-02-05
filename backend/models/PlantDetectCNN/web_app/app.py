from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import io
import base64
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class PlantDiseasePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = []
        self.img_size = 256
        self.load_model(model_path)
        self.load_class_names()
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                # Create model architecture for demo
                from tensorflow.keras.applications import MobileNetV2
                from tensorflow.keras import layers, models
                from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
                
                base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                        input_shape=(self.img_size, self.img_size, 3))
                base_model.trainable = False
                
                self.model = models.Sequential([
                    base_model,
                    GlobalAveragePooling2D(),
                    BatchNormalization(),
                    layers.Dense(256, activation='relu'),
                    Dropout(0.5),
                    layers.Dense(38, activation='softmax')  # 38 classes
                ])
                
                self.model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
                logger.info("Demo model architecture created")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def load_class_names(self):
        """Load class names for plant diseases"""
        # Common plant disease classes
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize((self.img_size, self.img_size))
            
            # Convert to array and normalize
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, "Error preprocessing image"
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]
            
            results = []
            for i, idx in enumerate(top_indices):
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
                confidence = float(predictions[0][idx])
                
                # Clean up class name for display
                display_name = class_name.replace('___', ' - ').replace('_', ' ')
                
                results.append({
                    'rank': i + 1,
                    'class': display_name,
                    'confidence': confidence,
                    'percentage': confidence * 100
                })
            
            return results, None
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None, f"Prediction error: {str(e)}"

# Initialize predictor
predictor = PlantDiseasePredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read image
            image = Image.open(io.BytesIO(file.read()))
            
            # Make prediction
            results, error = predictor.predict(image)
            
            if error:
                return jsonify({'error': error}), 500
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'predictions': results,
                'image': img_str,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch file upload"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        for file in files[:10]:  # Limit to 10 files
            if file and allowed_file(file.filename):
                try:
                    image = Image.open(io.BytesIO(file.read()))
                    predictions, error = predictor.predict(image)
                    
                    if error:
                        results.append({
                            'filename': file.filename,
                            'error': error
                        })
                    else:
                        results.append({
                            'filename': file.filename,
                            'predictions': predictions,
                            'top_prediction': predictions[0] if predictions else None
                        })
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        return jsonify({'error': f'Batch upload failed: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if predictor.model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_loaded': True,
            'input_shape': predictor.model.input_shape,
            'output_shape': predictor.model.output_shape,
            'total_parameters': predictor.model.count_params(),
            'num_classes': len(predictor.class_names),
            'class_names': predictor.class_names[:10],  # First 10 classes
            'image_size': predictor.img_size
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'timestamp': datetime.now().isoformat()
    })

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)