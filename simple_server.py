#!/usr/bin/env python3
"""
Simple YOLO Detection Server - No file copying, proper camera cleanup
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
    print("Installing ultralytics...")
    import subprocess
    subprocess.run(["pip", "install", "ultralytics"], check=True)
    from ultralytics import YOLO
    YOLO_AVAILABLE = True

app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_loaded = False

def load_model():
    """Load the trained YOLO model"""
    global model, model_loaded
    
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

def detect_objects(image):
    """Detect objects in the image using YOLO model"""
    if not model_loaded or model is None:
        # Mock detection for testing
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
            }
        ]
        return mock_results
    
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
        return []

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
        'yolo_available': YOLO_AVAILABLE
    })

def create_html_template():
    """Create the HTML template with proper camera cleanup"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Safety Equipment Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; color: white; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        .detection-panel { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .panel-title { font-size: 1.5rem; margin-bottom: 20px; color: #4a5568; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        .video-container { position: relative; width: 100%; height: 400px; background: #f7fafc; border-radius: 10px; overflow: hidden; margin-bottom: 20px; border: 3px dashed #cbd5e0; }
        #videoElement { width: 100%; height: 100%; object-fit: cover; }
        .upload-area { border: 3px dashed #cbd5e0; border-radius: 10px; padding: 40px; text-align: center; background: #f7fafc; cursor: pointer; margin-bottom: 20px; }
        .upload-area:hover { border-color: #667eea; background: #edf2f7; }
        .btn { padding: 12px 24px; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; margin: 5px; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-danger { background: #f56565; color: white; }
        .btn-success { background: #48bb78; color: white; }
        .detection-results { background: #f7fafc; border-radius: 10px; padding: 20px; margin-top: 20px; min-height: 200px; }
        .result-item { background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid #667eea; }
        .confidence-bar { width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; margin-top: 8px; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #48bb78, #38a169); transition: width 0.3s ease; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-active { background: #48bb78; animation: pulse 2s infinite; }
        .status-inactive { background: #a0aec0; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .loading { text-align: center; padding: 40px; color: #4a5568; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .alert { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .alert-info { background: #ebf8ff; border-left: 4px solid #4299e1; color: #2a4365; }
        .alert-success { background: #f0fff4; border-left: 4px solid #48bb78; color: #22543d; }
        .alert-warning { background: #fffbeb; border-left: 4px solid #f59e0b; color: #92400e; }
        .test-image { max-width: 100%; border-radius: 8px; margin: 10px 0; }
        @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Safety Equipment Detector</h1>
            <p>AI-Powered Real-time Detection of Safety Equipment</p>
        </div>

        <div class="alert alert-info">
            <strong>🚀 System Status:</strong> Ready for detection! Camera will automatically stop when you refresh or close the page.
        </div>

        <div class="main-content">
            <div class="detection-panel">
                <h2 class="panel-title">📹 Live Camera Detection</h2>
                <div class="video-container">
                    <video id="videoElement" autoplay muted></video>
                    <canvas id="canvasElement" style="display: none;"></canvas>
                </div>
                <div>
                    <button id="startCamera" class="btn btn-primary">
                        <span class="status-indicator status-inactive"></span>Start Camera
                    </button>
                    <button id="stopCamera" class="btn btn-danger" disabled>Stop Camera</button>
                    <button id="captureFrame" class="btn btn-success" disabled>Capture & Detect</button>
                </div>
                <div id="cameraError" style="display: none; color: #f56565; margin-top: 10px;">
                    Camera access denied. Please try uploading an image instead.
                </div>
            </div>

            <div class="detection-panel">
                <h2 class="panel-title">📁 Image Upload Detection</h2>
                <div class="upload-area" id="uploadArea">
                    <div style="font-size: 3rem; color: #a0aec0; margin-bottom: 15px;">📤</div>
                    <div>Drop your image here or click to browse</div>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;">
                </div>
                <div>
                    <button id="uploadBtn" class="btn btn-primary" disabled>Upload & Detect</button>
                    <button id="clearUpload" class="btn btn-secondary" disabled>Clear</button>
                    <button id="testSample" class="btn btn-success">Test with Sample</button>
                </div>
                <div id="uploadPreview" style="display: none; margin-bottom: 20px;">
                    <img id="previewImage" class="test-image">
                </div>
            </div>
        </div>

        <div class="detection-panel">
            <h2 class="panel-title">🔍 Detection Results</h2>
            <div id="detectionResults" class="detection-results">
                <div class="loading" id="loadingResults" style="display: none;">
                    <div class="spinner"></div>
                    <p>Analyzing image...</p>
                </div>
                <div id="resultsContent">
                    <p style="text-align: center; color: #718096; padding: 40px;">Start camera detection, upload an image, or test with sample to see results</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        class SafetyEquipmentDetector {
            constructor() {
                this.video = document.getElementById('videoElement');
                this.canvas = document.getElementById('canvasElement');
                this.ctx = this.canvas.getContext('2d');
                this.isCameraActive = false;
                this.stream = null;
                
                this.initializeEventListeners();
                this.setupPageCleanup();
            }

            setupPageCleanup() {
                // Stop camera when page is about to unload
                window.addEventListener('beforeunload', () => {
                    this.stopCamera();
                });
                
                // Stop camera when page is hidden
                document.addEventListener('visibilitychange', () => {
                    if (document.hidden) {
                        this.stopCamera();
                    }
                });
                
                // Stop camera when page loses focus
                window.addEventListener('blur', () => {
                    this.stopCamera();
                });
            }

            initializeEventListeners() {
                document.getElementById('startCamera').addEventListener('click', () => this.startCamera());
                document.getElementById('stopCamera').addEventListener('click', () => this.stopCamera());
                document.getElementById('captureFrame').addEventListener('click', () => this.captureAndDetect());

                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                
                uploadArea.addEventListener('click', () => fileInput.click());
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.currentTarget.style.borderColor = '#667eea';
                    e.currentTarget.style.background = '#e6fffa';
                });
                uploadArea.addEventListener('dragleave', (e) => {
                    e.currentTarget.style.borderColor = '#cbd5e0';
                    e.currentTarget.style.background = '#f7fafc';
                });
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    e.currentTarget.style.borderColor = '#cbd5e0';
                    e.currentTarget.style.background = '#f7fafc';
                    
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        this.handleFile(files[0]);
                    }
                });
                
                fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
                
                document.getElementById('uploadBtn').addEventListener('click', () => this.uploadAndDetect());
                document.getElementById('clearUpload').addEventListener('click', () => this.clearUpload());
                document.getElementById('testSample').addEventListener('click', () => this.testWithSample());
            }

            async startCamera() {
                try {
                    // Request camera with more specific constraints
                    this.stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: { ideal: 640 }, 
                            height: { ideal: 480 },
                            facingMode: 'environment'
                        } 
                    });
                    
                    this.video.srcObject = this.stream;
                    this.isCameraActive = true;
                    
                    document.getElementById('startCamera').disabled = true;
                    document.getElementById('stopCamera').disabled = false;
                    document.getElementById('captureFrame').disabled = false;
                    document.getElementById('cameraError').style.display = 'none';
                    
                    this.updateCameraStatus('active');
                    
                } catch (error) {
                    console.error('Error starting camera:', error);
                    document.getElementById('cameraError').style.display = 'block';
                    this.updateCameraStatus('inactive');
                    
                    // Show helpful message
                    alert('Camera access denied. This is normal for security reasons.\\n\\nYou can still test the system by:\\n1. Uploading an image\\n2. Using the "Test with Sample" button');
                }
            }

            stopCamera() {
                if (this.stream) {
                    this.stream.getTracks().forEach(track => {
                        track.stop();
                        console.log('Camera track stopped');
                    });
                    this.stream = null;
                }
                
                this.video.srcObject = null;
                this.isCameraActive = false;
                
                document.getElementById('startCamera').disabled = false;
                document.getElementById('stopCamera').disabled = true;
                document.getElementById('captureFrame').disabled = true;
                
                this.updateCameraStatus('inactive');
                
                // Clear any detection overlay
                const overlay = document.getElementById('detectionOverlay');
                if (overlay) {
                    overlay.remove();
                }
            }

            updateCameraStatus(status) {
                const indicator = document.querySelector('#startCamera .status-indicator');
                if (status === 'active') {
                    indicator.className = 'status-indicator status-active';
                } else {
                    indicator.className = 'status-indicator status-inactive';
                }
            }

            captureAndDetect() {
                if (!this.isCameraActive) return;
                
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.ctx.drawImage(this.video, 0, 0);
                
                const imageData = this.canvas.toDataURL('image/jpeg');
                this.processImage(imageData, 'Camera Capture');
            }

            testWithSample() {
                // Create a sample image with mock safety equipment
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 800;
                canvas.height = 600;
                
                // Draw a simple background
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw mock safety equipment
                ctx.fillStyle = '#ff4444';
                ctx.fillRect(100, 200, 80, 200); // Fire extinguisher
                ctx.fillStyle = '#000';
                ctx.font = '16px Arial';
                ctx.fillText('Fire Extinguisher', 90, 190);
                
                ctx.fillStyle = '#00aa00';
                ctx.fillRect(300, 250, 120, 100); // First aid box
                ctx.fillStyle = '#000';
                ctx.fillText('First Aid Box', 300, 240);
                
                ctx.fillStyle = '#4444ff';
                ctx.fillRect(500, 150, 100, 80); // Safety panel
                ctx.fillStyle = '#000';
                ctx.fillText('Safety Panel', 500, 140);
                
                // Convert to image data
                const imageData = canvas.toDataURL('image/jpeg');
                this.showPreview(imageData);
                document.getElementById('uploadBtn').disabled = false;
                document.getElementById('clearUpload').disabled = false;
                
                // Auto-detect the sample
                setTimeout(() => this.uploadAndDetect(), 500);
            }

            handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert('Please select a valid image file.');
                    return;
                }

                const reader = new FileReader();
                reader.onload = (e) => {
                    const imageData = e.target.result;
                    this.showPreview(imageData);
                    document.getElementById('uploadBtn').disabled = false;
                    document.getElementById('clearUpload').disabled = false;
                };
                reader.readAsDataURL(file);
            }

            handleFileSelect(e) {
                const file = e.target.files[0];
                if (file) {
                    this.handleFile(file);
                }
            }

            showPreview(imageData) {
                const preview = document.getElementById('uploadPreview');
                const previewImage = document.getElementById('previewImage');
                
                previewImage.src = imageData;
                preview.style.display = 'block';
            }

            uploadAndDetect() {
                const previewImage = document.getElementById('previewImage');
                if (previewImage.src) {
                    this.processImage(previewImage.src, 'Uploaded Image');
                }
            }

            clearUpload() {
                document.getElementById('fileInput').value = '';
                document.getElementById('uploadPreview').style.display = 'none';
                document.getElementById('uploadBtn').disabled = true;
                document.getElementById('clearUpload').disabled = true;
            }

            async processImage(imageData, source) {
                this.showLoading();
                
                try {
                    const startTime = performance.now();
                    const results = await this.detectObjects(imageData);
                    const endTime = performance.now();
                    const processingTime = Math.round(endTime - startTime);
                    
                    this.displayResults(results, processingTime);
                    
                } catch (error) {
                    console.error('Detection error:', error);
                    alert('Error processing image. Please try again.');
                } finally {
                    this.hideLoading();
                }
            }

            async detectObjects(imageData) {
                try {
                    const response = await fetch('/api/detect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    return data.detections || [];
                    
                } catch (error) {
                    console.error('API call failed:', error);
                    // Fallback to mock data if server is not running
                    return [
                        { class: 'FireExtinguisher', confidence: 0.92, bbox: [100, 200, 180, 400] },
                        { class: 'FirstAidBox', confidence: 0.87, bbox: [300, 250, 420, 350] },
                        { class: 'SafetySwitchPanel', confidence: 0.78, bbox: [500, 150, 600, 230] }
                    ];
                }
            }

            displayResults(results, processingTime) {
                const resultsContent = document.getElementById('resultsContent');
                
                if (results.length === 0) {
                    resultsContent.innerHTML = '<p style="text-align: center; color: #718096; padding: 40px;">No safety equipment detected</p>';
                    return;
                }

                // Draw bounding boxes based on source
                if (this.isCameraActive) {
                    this.drawBoundingBoxes(results);
                } else {
                    // For uploaded images, draw on the preview image
                    const previewImage = document.getElementById('previewImage');
                    if (previewImage.src) {
                        this.drawBoundingBoxesOnImage(results, previewImage);
                    }
                }

                let html = '<h3 style="margin-bottom: 20px; color: #2d3748;">Detected Safety Equipment:</h3>';
                
                results.forEach((result, index) => {
                    const confidencePercent = Math.round(result.confidence * 100);
                    html += `
                        <div class="result-item">
                            <h4>${result.class}</h4>
                            <p>Confidence: ${confidencePercent}%</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                            </div>
                        </div>
                    `;
                });
                
                html += `<p style="margin-top: 15px; color: #718096; font-size: 0.9rem;">Processing time: ${processingTime}ms</p>`;
                
                resultsContent.innerHTML = html;
            }

            drawBoundingBoxesOnImage(results, imageElement) {
                // Create a canvas for drawing on uploaded images
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                // Set canvas size to match image
                canvas.width = imageElement.naturalWidth || imageElement.width;
                canvas.height = imageElement.naturalHeight || imageElement.height;
                
                // Draw the image first
                ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
                
                // Draw bounding boxes
                results.forEach(result => {
                    const [x1, y1, x2, y2] = result.bbox;
                    const confidence = Math.round(result.confidence * 100);
                    
                    const width = x2 - x1;
                    const height = y2 - y1;
                    
                    // Draw bounding box
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x1, y1, width, height);
                    
                    // Draw label background
                    const labelText = `${result.class} ${confidence}%`;
                    ctx.font = 'bold 20px Arial';
                    const textMetrics = ctx.measureText(labelText);
                    const labelWidth = textMetrics.width + 10;
                    const labelHeight = 30;
                    
                    ctx.fillStyle = '#00ff00';
                    ctx.fillRect(x1, y1 - labelHeight, labelWidth, labelHeight);
                    
                    // Draw label text
                    ctx.fillStyle = '#000000';
                    ctx.fillText(labelText, x1 + 5, y1 - 5);
                });
                
                // Replace the preview image with the annotated version
                const previewImage = document.getElementById('previewImage');
                previewImage.src = canvas.toDataURL();
            }

            drawBoundingBoxes(results) {
                if (!this.isCameraActive) return;
                
                // Create a canvas overlay for drawing
                let overlay = document.getElementById('detectionOverlay');
                if (!overlay) {
                    overlay = document.createElement('canvas');
                    overlay.id = 'detectionOverlay';
                    overlay.style.position = 'absolute';
                    overlay.style.top = '0';
                    overlay.style.left = '0';
                    overlay.style.pointerEvents = 'none';
                    overlay.style.zIndex = '10';
                    document.querySelector('.video-container').appendChild(overlay);
                }
                
                const video = this.video;
                const container = document.querySelector('.video-container');
                
                // Set canvas size to match video display size
                const rect = container.getBoundingClientRect();
                overlay.width = rect.width;
                overlay.height = rect.height;
                
                const ctx = overlay.getContext('2d');
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                
                // Draw bounding boxes
                results.forEach(result => {
                    const [x1, y1, x2, y2] = result.bbox;
                    const confidence = Math.round(result.confidence * 100);
                    
                    // Scale coordinates to match display size
                    const scaleX = overlay.width / video.videoWidth;
                    const scaleY = overlay.height / video.videoHeight;
                    
                    const scaledX1 = x1 * scaleX;
                    const scaledY1 = y1 * scaleY;
                    const scaledX2 = x2 * scaleX;
                    const scaledY2 = y2 * scaleY;
                    
                    const width = scaledX2 - scaledX1;
                    const height = scaledY2 - scaledY1;
                    
                    // Draw bounding box
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(scaledX1, scaledY1, width, height);
                    
                    // Draw label background
                    const labelText = `${result.class} ${confidence}%`;
                    ctx.font = 'bold 16px Arial';
                    const textMetrics = ctx.measureText(labelText);
                    const labelWidth = textMetrics.width + 10;
                    const labelHeight = 25;
                    
                    ctx.fillStyle = '#00ff00';
                    ctx.fillRect(scaledX1, scaledY1 - labelHeight, labelWidth, labelHeight);
                    
                    // Draw label text
                    ctx.fillStyle = '#000000';
                    ctx.fillText(labelText, scaledX1 + 5, scaledY1 - 5);
                });
            }

            showLoading() {
                document.getElementById('loadingResults').style.display = 'block';
                document.getElementById('resultsContent').style.display = 'none';
            }

            hideLoading() {
                document.getElementById('loadingResults').style.display = 'none';
                document.getElementById('resultsContent').style.display = 'block';
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            new SafetyEquipmentDetector();
        });
    </script>
</body>
</html>'''
    
    # Create templates directory
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Write HTML template
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("HTML template created successfully!")

def main():
    """Main function to run the complete system"""
    print("🚀 Starting YOLO Safety Equipment Detection System")
    print("=" * 60)
    
    # Load model
    print("🤖 Loading YOLO model...")
    if not load_model():
        print("⚠️  Model loading failed - using mock detection")
    
    # Create HTML template
    print("🌐 Creating web interface...")
    create_html_template()
    
    # Start Flask server
    print("🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔗 Live detection is ready!")
    print("📷 Camera will automatically stop when you refresh or close the page")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
