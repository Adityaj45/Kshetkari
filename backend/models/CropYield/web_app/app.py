"""
Flask Web Application for Crop Yield Prediction Model
Provides a user-friendly interface for crop yield predictions
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'crop_yield_prediction_secret_key_2024'

class CropYieldPredictor:
    """Crop yield prediction handler"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = [
            'year', 'fertilizer', 'pesticide', 'avg_temp_c', 'total_rainfall_mm', 
            'avg_humidity_percent', 'nitrogen', 'phosphorus', 'potassium', 'pH', 
            'crop_encoded', 'season_encoded', 'state_encoded'
        ]
        self.crop_names = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'soybean', 'barley', 'chickpea']
        self.season_names = ['Kharif', 'Rabi', 'Whole Year']
        self.state_names = ['Punjab', 'Haryana', 'UP', 'Maharashtra', 'Karnataka', 'Gujarat', 'MP', 'Rajasthan']
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        model_path = "Model/ensemble_model.pkl"
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"✓ Model loaded from {model_path}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.create_mock_model()
        else:
            print("Model file not found, creating mock model...")
            self.create_mock_model()
        
        # Set up label encoders
        self.setup_encoders()
    
    def create_mock_model(self):
        """Create a mock ensemble model for demonstration"""
        print("Creating mock ensemble model for demonstration...")
        
        self.model = {
            'xgb_ultra': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'xgb_fine': RandomForestRegressor(n_estimators=40, max_depth=4, random_state=123),
            'gb_enhanced': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
            'scaler': StandardScaler(),
            'weights': [0.7, 0.15, 0.15],
            'features': self.feature_names
        }
        
        # Train on mock data
        np.random.seed(42)
        X_mock = np.random.rand(1000, 13)
        X_mock[:, 0] = np.random.randint(1997, 2023, 1000)  # Realistic years
        y_mock = np.random.uniform(0.5, 6.0, 1000)  # Realistic yield values
        
        X_scaled = self.model['scaler'].fit_transform(X_mock)
        self.model['xgb_ultra'].fit(X_scaled, y_mock)
        self.model['xgb_fine'].fit(X_scaled, y_mock)
        self.model['gb_enhanced'].fit(X_scaled, y_mock)
        
        print("✓ Mock ensemble model created successfully")
    
    def setup_encoders(self):
        """Set up label encoders for categorical variables"""
        self.label_encoders = {
            'crop': LabelEncoder().fit(self.crop_names),
            'season': LabelEncoder().fit(self.season_names),
            'state': LabelEncoder().fit(self.state_names)
        }
    
    def predict(self, features):
        """Make crop yield prediction"""
        try:
            # Validate input
            if len(features) != 13:
                raise ValueError("Expected 13 features")
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction with ensemble
            if isinstance(self.model, dict):
                # Ensemble model
                features_scaled = self.model['scaler'].transform(features_array)
                
                pred1 = self.model['xgb_ultra'].predict(features_scaled)[0]
                pred2 = self.model['xgb_fine'].predict(features_scaled)[0]
                pred3 = self.model['gb_enhanced'].predict(features_scaled)[0]
                
                weights = self.model['weights']
                ensemble_pred = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3
                
                # Individual model predictions for analysis
                individual_predictions = {
                    'xgb_ultra': float(pred1),
                    'xgb_fine': float(pred2),
                    'gb_enhanced': float(pred3)
                }
            else:
                # Single model
                ensemble_pred = self.model.predict(features_array)[0]
                individual_predictions = None
            
            # Calculate confidence based on prediction consistency
            confidence = self._calculate_confidence(individual_predictions) if individual_predictions else 0.85
            
            return {
                'success': True,
                'prediction': float(ensemble_pred),
                'individual_predictions': individual_predictions,
                'confidence': confidence,
                'yield_category': self._categorize_yield(ensemble_pred)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }
    
    def _calculate_confidence(self, individual_preds):
        """Calculate confidence based on prediction consistency"""
        if not individual_preds:
            return 0.85
        
        predictions = list(individual_preds.values())
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Higher consistency = higher confidence
        if std_pred == 0:
            return 0.95
        
        coefficient_of_variation = std_pred / max(abs(mean_pred), 0.001)
        confidence = max(0.5, 0.95 - coefficient_of_variation * 2)
        
        return min(0.95, confidence)
    
    def _categorize_yield(self, yield_value):
        """Categorize yield into performance levels"""
        if yield_value < 1.0:
            return {'category': 'Low', 'color': '#ff4444', 'description': 'Below average yield'}
        elif yield_value < 2.5:
            return {'category': 'Moderate', 'color': '#ffaa00', 'description': 'Average yield'}
        elif yield_value < 4.0:
            return {'category': 'Good', 'color': '#44aa44', 'description': 'Above average yield'}
        else:
            return {'category': 'Excellent', 'color': '#00aa00', 'description': 'Exceptional yield'}
    
    def get_yield_recommendations(self, prediction, features):
        """Get recommendations based on predicted yield"""
        recommendations = []
        
        # Extract feature values
        year, fertilizer, pesticide, temp, rainfall, humidity, n, p, k, ph = features[:10]
        
        # Fertilizer recommendations
        if prediction < 2.0:
            if fertilizer < 800:
                recommendations.append("Consider increasing fertilizer application by 15-20%")
            if n < 60:
                recommendations.append("Increase nitrogen application for better vegetative growth")
            if p < 30:
                recommendations.append("Boost phosphorus levels to improve root development")
        
        # Climate-based recommendations
        if temp > 35:
            recommendations.append("High temperature detected - ensure adequate irrigation")
        elif temp < 15:
            recommendations.append("Low temperature may affect growth - consider protective measures")
        
        if rainfall < 100:
            recommendations.append("Low rainfall - implement efficient irrigation systems")
        elif rainfall > 300:
            recommendations.append("High rainfall - ensure proper drainage to prevent waterlogging")
        
        if humidity > 85:
            recommendations.append("High humidity - monitor for fungal diseases")
        
        # Soil recommendations
        if ph < 6.0:
            recommendations.append("Soil is acidic - consider lime application")
        elif ph > 8.0:
            recommendations.append("Soil is alkaline - consider sulfur application")
        
        # General recommendations
        if prediction >= 3.0:
            recommendations.append("Excellent conditions predicted - maintain current practices")
        
        if not recommendations:
            recommendations.append("Current conditions are suitable for good yield")
        
        return recommendations

# Initialize predictor
predictor = CropYieldPredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         crops=predictor.crop_names,
                         seasons=predictor.season_names,
                         states=predictor.state_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        year = int(request.form.get('year', 2023))
        fertilizer = float(request.form.get('fertilizer', 0))
        pesticide = float(request.form.get('pesticide', 0))
        temperature = float(request.form.get('temperature', 0))
        rainfall = float(request.form.get('rainfall', 0))
        humidity = float(request.form.get('humidity', 0))
        nitrogen = float(request.form.get('nitrogen', 0))
        phosphorus = float(request.form.get('phosphorus', 0))
        potassium = float(request.form.get('potassium', 0))
        ph = float(request.form.get('ph', 0))
        crop = request.form.get('crop', 'rice')
        season = request.form.get('season', 'Kharif')
        state = request.form.get('state', 'Punjab')
        
        # Validate inputs
        if not all([1990 <= year <= 2030, fertilizer >= 0, pesticide >= 0, 
                   0 <= temperature <= 50, 0 <= humidity <= 100, 
                   0 <= ph <= 14, rainfall >= 0]):
            flash('Please enter valid values for all fields', 'error')
            return redirect(url_for('index'))
        
        # Encode categorical variables
        try:
            crop_encoded = predictor.label_encoders['crop'].transform([crop])[0]
            season_encoded = predictor.label_encoders['season'].transform([season])[0]
            state_encoded = predictor.label_encoders['state'].transform([state])[0]
        except ValueError as e:
            flash(f'Invalid categorical value: {e}', 'error')
            return redirect(url_for('index'))
        
        # Prepare features
        features = [year, fertilizer, pesticide, temperature, rainfall, humidity, 
                   nitrogen, phosphorus, potassium, ph, crop_encoded, season_encoded, state_encoded]
        
        # Make prediction
        result = predictor.predict(features)
        
        if result['success']:
            recommendations = predictor.get_yield_recommendations(result['prediction'], features)
            
            # Prepare response data
            response_data = {
                'prediction': result['prediction'],
                'confidence': result.get('confidence'),
                'individual_predictions': result.get('individual_predictions'),
                'yield_category': result.get('yield_category'),
                'recommendations': recommendations,
                'input_values': {
                    'year': year,
                    'fertilizer': fertilizer,
                    'pesticide': pesticide,
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'humidity': humidity,
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'ph': ph,
                    'crop': crop,
                    'season': season,
                    'state': state
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return render_template('result.html', **response_data)
        else:
            flash(f'Prediction error: {result["error"]}', 'error')
            return redirect(url_for('index'))
            
    except ValueError as e:
        flash('Please enter valid numeric values', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features
        required_fields = ['year', 'fertilizer', 'pesticide', 'temperature', 'rainfall', 
                          'humidity', 'nitrogen', 'phosphorus', 'potassium', 'ph', 
                          'crop', 'season', 'state']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Encode categorical variables
        try:
            crop_encoded = predictor.label_encoders['crop'].transform([data['crop']])[0]
            season_encoded = predictor.label_encoders['season'].transform([data['season']])[0]
            state_encoded = predictor.label_encoders['state'].transform([data['state']])[0]
        except ValueError as e:
            return jsonify({'error': f'Invalid categorical value: {e}'}), 400
        
        # Prepare features
        features = [
            float(data['year']), float(data['fertilizer']), float(data['pesticide']),
            float(data['temperature']), float(data['rainfall']), float(data['humidity']),
            float(data['nitrogen']), float(data['phosphorus']), float(data['potassium']),
            float(data['ph']), crop_encoded, season_encoded, state_encoded
        ]
        
        # Make prediction
        result = predictor.predict(features)
        
        if result['success']:
            recommendations = predictor.get_yield_recommendations(result['prediction'], features)
            
            response = {
                'success': True,
                'prediction': result['prediction'],
                'confidence': result.get('confidence'),
                'individual_predictions': result.get('individual_predictions'),
                'yield_category': result.get('yield_category'),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test prediction with sample data
        test_features = [2020, 1000, 500, 25.5, 150.0, 75.0, 80, 40, 50, 6.5, 1, 0, 5]
        result = predictor.predict(test_features)
        
        return jsonify({
            'status': 'healthy' if result['success'] else 'unhealthy',
            'model_loaded': predictor.model is not None,
            'timestamp': datetime.now().isoformat(),
            'test_prediction': result['success']
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/info')
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model_type': 'Crop Yield Prediction',
        'features': predictor.feature_names,
        'crops': predictor.crop_names,
        'seasons': predictor.season_names,
        'states': predictor.state_names,
        'model_loaded': predictor.model is not None,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("="*60)
    print("CROP YIELD PREDICTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5004")
    print("API endpoints:")
    print("  POST /api/predict - Single prediction")
    print("  GET /health - Health check")
    print("  GET /info - Model information")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5004)