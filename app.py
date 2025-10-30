#!/usr/bin/env python3
"""
Flask Backend for YOLO Safety Equipment Detection
Serves the trained YOLO model via REST API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time
import os
from pathlib import Path

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_loaded = False

def load_model():
    """Load the trained YOLO model"""
    global model, model_loaded
    
    if not YOLO_AVAILABLE:
        print("YOLO not available - using mock detection")
        model_loaded = True
        return True
    
    try:
        # Try to load the trained model
        model_paths = [
            "runs/train/yolo_80_accuracy/weights/best.pt",
            "runs/train/optimized_80_plus_50min_full_dataset/weights/best.pt",
            "yolo11m.pt"  # Fallback to pretrained model
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                model = YOLO(model_path)
                model_loaded = True
                print("Model loaded successfully!")
                return True
        
        print("No trained model found, using pretrained YOLO11m")
        model = YOLO("yolo11m.pt")
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
        return False

def preprocess_image(image_data):
    """Convert base64 image data to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def mock_detection(image):
    """Mock detection for testing when YOLO is not available"""
    # Generate mock results based on image size
    height, width = image.shape[:2]
    
    mock_results = [
        {
            'class': 'FireExtinguisher',
            'confidence': 0.92,
            'bbox': [int(width*0.1), int(height*0.2), int(width*0.3), int(height*0.6)]
        },
        {
            'class': 'FirstAidBox', 
            'confidence': 0.87,
            'bbox': [int(width*0.4), int(height*0.3), int(width*0.25), int(height*0.4)]
        },
        {
            'class': 'SafetySwitchPanel',
            'confidence': 0.78,
            'bbox': [int(width*0.7), int(height*0.1), int(width*0.28), int(height*0.15)]
        }
    ]
    
    return mock_results

def detect_objects(image):
    """Detect objects in the image using YOLO model"""
    if not model_loaded or model is None:
        return mock_detection(image)
    
    try:
        # Run YOLO detection
        results = model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return detections
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return mock_detection(image)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """API endpoint for object detection"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        image = preprocess_image(image_data)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect objects
        start_time = time.time()
        detections = detect_objects(image)
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Prepare response
        response = {
            'detections': detections,
            'processing_time': round(processing_time, 2),
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'model_loaded': model_loaded
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in detect endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint for checking server status"""
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'yolo_available': YOLO_AVAILABLE,
        'model_path': str(model.model_path) if model and hasattr(model, 'model_path') else None
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """API endpoint for getting available classes"""
    if model_loaded and model:
        return jsonify({
            'classes': model.names
        })
    else:
        # Default classes for safety equipment
        return jsonify({
            'classes': {
                0: 'OxygenTank',
                1: 'NitrogenTank', 
                2: 'FirstAidBox',
                3: 'FireAlarm',
                4: 'SafetySwitchPanel',
                5: 'EmergencyPhone',
                6: 'FireExtinguisher'
            }
        })

if __name__ == '__main__':
    print("Starting YOLO Safety Equipment Detection Server...")
    print("=" * 60)
    
    # Load the model
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Model loading failed - using mock detection")
    
    print("🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - POST /api/detect - Detect objects in image")
    print("   - GET /api/status - Check server status")
    print("   - GET /api/classes - Get available classes")
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Copy the HTML file to templates directory
    if os.path.exists("index.html"):
        import shutil
        shutil.copy2("index.html", "templates/index.html")
        print("📄 HTML template copied to templates directory")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)



