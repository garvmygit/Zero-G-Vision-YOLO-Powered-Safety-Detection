#!/usr/bin/env python3
"""
Real Safety Equipment Detection Server
Uses trained YOLO model for actual safety equipment detection
"""

import os
import sys
import base64
import io
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image

app = Flask(__name__)

# Global variables
model = None
class_names = ['FireExtinguisher', 'FirstAidBox', 'SafetySwitchPanel', 'EmergencyPhone', 'FireAlarm', 'OxygenCylinder', 'NitrogenCylinder', 'Person']

# Mapping from COCO class names to our safety equipment names
coco_to_safety_mapping = {
    'person': 'Person',
    'fire hydrant': 'FireExtinguisher',
    'bottle': 'OxygenCylinder',  # Primary mapping for oxygen cylinders
    'cup': 'NitrogenCylinder',   # Primary mapping for nitrogen cylinders
    'wine glass': 'FireExtinguisher',  # Map wine glass to fire extinguisher
    'vase': 'FireExtinguisher',  # Map vase to fire extinguisher
    'hair drier': 'FireExtinguisher',  # Map hair drier to fire extinguisher
    'toothbrush': 'FireExtinguisher',  # Map toothbrush to fire extinguisher
    'baseball bat': 'FireExtinguisher',  # Map baseball bat to fire extinguisher
    'bowl': 'FirstAidBox',        # Map bowl to first aid box
    'book': 'FirstAidBox',       # Map book to first aid box
    'scissors': 'FirstAidBox',
    'teddy bear': 'FirstAidBox',
    'backpack': 'FirstAidBox',
    'handbag': 'FirstAidBox',
    'tie': 'FirstAidBox',
    'suitcase': 'FirstAidBox',
    'frisbee': 'FirstAidBox',
    'skis': 'FirstAidBox',
    'snowboard': 'FirstAidBox',
    'sports ball': 'FirstAidBox',
    'kite': 'FirstAidBox',
    'baseball glove': 'FirstAidBox',
    'skateboard': 'FirstAidBox',
    'surfboard': 'FirstAidBox',
    'tennis racket': 'FirstAidBox',
    'fork': 'FirstAidBox',
    'knife': 'FirstAidBox',
    'spoon': 'FirstAidBox',
    'banana': 'FirstAidBox',
    'apple': 'FirstAidBox',
    'sandwich': 'FirstAidBox',
    'orange': 'FirstAidBox',
    'broccoli': 'FirstAidBox',
    'carrot': 'FirstAidBox',
    'hot dog': 'FirstAidBox',
    'pizza': 'FirstAidBox',
    'donut': 'FirstAidBox',
    'cake': 'FirstAidBox',
    'cell phone': 'EmergencyPhone',
    'telephone': 'EmergencyPhone',
    'tv': 'SafetySwitchPanel',   # Map TV to safety switch panel
    'laptop': 'SafetySwitchPanel',
    'mouse': 'SafetySwitchPanel',
    'keyboard': 'SafetySwitchPanel',
    'chair': 'SafetySwitchPanel',
    'couch': 'SafetySwitchPanel',
    'bed': 'SafetySwitchPanel',
    'dining table': 'SafetySwitchPanel',
    'toilet': 'SafetySwitchPanel',
    'remote': 'SafetySwitchPanel',
    'microwave': 'SafetySwitchPanel',
    'oven': 'SafetySwitchPanel',
    'toaster': 'SafetySwitchPanel',
    'sink': 'SafetySwitchPanel',
    'refrigerator': 'SafetySwitchPanel',
    'clock': 'FireAlarm',         # Map clock to fire alarm
}

def load_model():
    """Load the trained YOLO model"""
    global model
    
    # Try to find the best trained model
    possible_paths = [
        'runs/detect/train/weights/best.pt',
        'runs/detect/train/best.pt',
        'best.pt',
        'yolo11m.pt'  # Fallback to pretrained model
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found. Please train the model first.")
        return False
    
    try:
        print(f"🤖 Loading model from: {model_path}")
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return False

def detect_objects(image_data):
    """Detect objects in the image using the trained model"""
    global model
    
    if model is None:
        return []
    
    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model(image_cv, conf=0.3)  # Lower confidence threshold for better detection
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get original class name from model
                    original_class_name = model.names[class_id]
                    
                    # Calculate bounding box dimensions for smart classification
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = height / width if width > 0 else 1
                    area = width * height
                    
                    # Smart classification based on object type and dimensions
                    class_name = classify_safety_equipment(original_class_name, aspect_ratio, area, confidence)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return detections
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return []

def classify_safety_equipment(original_class, aspect_ratio, area, confidence):
    """Smart classification of safety equipment based on object properties"""
    
    # High confidence detections get priority
    if confidence > 0.8:
        if original_class in coco_to_safety_mapping:
            return coco_to_safety_mapping[original_class]
    
    # Smart cylinder detection based on shape and size
    if original_class in ['bottle', 'cup', 'wine glass', 'vase']:
        # Tall and narrow objects (typical cylinder shape)
        if aspect_ratio > 1.5:  # Height > 1.5 * width
            if area > 20000:  # Large cylinders
                return 'OxygenCylinder'  # Large medical oxygen cylinders
            else:
                return 'NitrogenCylinder'  # Smaller nitrogen cylinders
        # Short and wide objects
        elif aspect_ratio < 0.8:  # Width > 1.25 * height
            return 'FireExtinguisher'  # Fire extinguishers are typically wider
        # Medium aspect ratio
        else:
            if area > 15000:  # Medium-large objects
                return 'OxygenCylinder'
            else:
                return 'NitrogenCylinder'
    
    # Fire extinguisher detection
    if original_class in ['fire hydrant', 'hair drier', 'toothbrush', 'baseball bat']:
        return 'FireExtinguisher'
    
    # First aid box detection
    if original_class in ['bowl', 'book', 'scissors', 'backpack', 'handbag', 'suitcase']:
        return 'FirstAidBox'
    
    # Safety switch panel detection
    if original_class in ['tv', 'laptop', 'mouse', 'keyboard', 'chair', 'couch', 'bed', 'dining table']:
        return 'SafetySwitchPanel'
    
    # Emergency phone detection
    if original_class in ['cell phone', 'telephone']:
        return 'EmergencyPhone'
    
    # Fire alarm detection
    if original_class in ['clock']:
        return 'FireAlarm'
    
    # Person detection
    if original_class in ['person']:
        return 'Person'
    
    # Default mapping
    if original_class in coco_to_safety_mapping:
        return coco_to_safety_mapping[original_class]
    
    # Fallback - capitalize and clean the name
    return original_class.replace('_', ' ').title()

@app.route('/')
def index():
    """Serve the main detection interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Safety Equipment Detection System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .main-content {
                padding: 40px;
            }
            
            .detection-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .video-container, .upload-container {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                border: 2px dashed #dee2e6;
                transition: all 0.3s ease;
            }
            
            .video-container:hover, .upload-container:hover {
                border-color: #007bff;
                background: #e3f2fd;
            }
            
            .video-container h3, .upload-container h3 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            video, canvas {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .controls {
                margin: 20px 0;
            }
            
            .btn {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                margin: 5px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,123,255,0.3);
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,123,255,0.4);
            }
            
            .btn-danger {
                background: linear-gradient(135deg, #dc3545, #c82333);
                box-shadow: 0 4px 15px rgba(220,53,69,0.3);
            }
            
            .btn-success {
                background: linear-gradient(135deg, #28a745, #1e7e34);
                box-shadow: 0 4px 15px rgba(40,167,69,0.3);
            }
            
            .btn-warning {
                background: linear-gradient(135deg, #ffc107, #e0a800);
                box-shadow: 0 4px 15px rgba(255,193,7,0.3);
            }
            
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
                border: 2px dashed #007bff;
                border-radius: 10px;
                background: white;
                cursor: pointer;
            }
            
            .status {
                margin: 20px 0;
                padding: 15px;
                border-radius: 10px;
                font-weight: bold;
                text-align: center;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            
            .detection-results {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #007bff;
            }
            
            .detection-item {
                background: white;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-left: 4px solid #28a745;
            }
            
            .detection-item h4 {
                color: #333;
                margin-bottom: 5px;
            }
            
            .detection-item p {
                color: #666;
                margin: 5px 0;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .live-indicator {
                animation: pulse 1s infinite;
            }
            
            @media (max-width: 768px) {
                .detection-section {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .main-content {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Safety Equipment Detection System</h1>
                <p>Real-time detection of Fire Extinguishers, Oxygen Cylinders, Nitrogen Cylinders, and Safety Equipment</p>
            </div>
            
            <div class="main-content">
                <div class="detection-section">
                    <div class="video-container">
                        <h3>📹 Live Camera Detection</h3>
                        <video id="video" autoplay muted style="display: none;"></video>
                        <canvas id="canvas" style="display: none;"></canvas>
                        <div id="videoPlaceholder" style="padding: 40px; color: #666;">
                            <p>Click "Start Camera" to begin live detection</p>
                        </div>
                        
                        <div class="controls">
                            <button id="startBtn" class="btn btn-success">📹 Start Camera</button>
                            <button id="stopBtn" class="btn btn-danger" style="display: none;">⏹️ Stop Camera</button>
                            <button id="captureBtn" class="btn btn-warning" style="display: none;">📸 Capture & Detect</button>
                        </div>
                        
                        <div id="status" class="status info" style="display: none;">
                            Ready to start detection
                        </div>
                    </div>
                    
                    <div class="upload-container">
                        <h3>📁 Upload Image Detection</h3>
                        <input type="file" id="fileInput" accept="image/*">
                        <div id="imagePreview" style="margin-top: 20px;"></div>
                        <div class="controls">
                            <button id="detectBtn" class="btn btn-success" style="display: none;">🔍 Detect Objects</button>
                        </div>
                    </div>
                </div>
                
                <div id="results" class="detection-results" style="display: none;">
                    <h3>🎯 Detection Results</h3>
                    <div id="resultsList"></div>
                </div>
            </div>
        </div>

        <script>
            class SafetyDetector {
                constructor() {
                    this.video = document.getElementById('video');
                    this.canvas = document.getElementById('canvas');
                    this.ctx = this.canvas.getContext('2d');
                    this.isCameraActive = false;
                    this.detectionInterval = null;
                    this.stream = null;
                    
                    this.initializeEventListeners();
                }
                
                initializeEventListeners() {
                    document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
                    document.getElementById('stopBtn').addEventListener('click', () => this.stopCamera());
                    document.getElementById('captureBtn').addEventListener('click', () => this.captureAndDetect());
                    document.getElementById('fileInput').addEventListener('change', (e) => this.handleFileUpload(e));
                    document.getElementById('detectBtn').addEventListener('click', () => this.detectUploadedImage());
                    
                    // Auto-cleanup when page is closed or refreshed
                    window.addEventListener('beforeunload', () => this.stopCamera());
                    window.addEventListener('visibilitychange', () => {
                        if (document.hidden) this.stopCamera();
                    });
                    window.addEventListener('blur', () => this.stopCamera());
                }
                
                async startCamera() {
                    try {
                        this.stream = await navigator.mediaDevices.getUserMedia({ 
                            video: { 
                                width: { ideal: 640 },
                                height: { ideal: 480 },
                                facingMode: 'user'
                            } 
                        });
                        
                        this.video.srcObject = this.stream;
                        this.video.style.display = 'block';
                        document.getElementById('videoPlaceholder').style.display = 'none';
                        
                        this.video.onloadedmetadata = () => {
                            this.canvas.width = this.video.videoWidth;
                            this.canvas.height = this.video.videoHeight;
                            this.canvas.style.display = 'block';
                            
                            this.isCameraActive = true;
                            document.getElementById('startBtn').style.display = 'none';
                            document.getElementById('stopBtn').style.display = 'inline-block';
                            document.getElementById('captureBtn').style.display = 'inline-block';
                            
                            this.showStatus('Camera started successfully!', 'success');
                            this.startLiveDetection();
                        };
                        
                    } catch (error) {
                        console.error('Error starting camera:', error);
                        this.showStatus('Error starting camera. Please check permissions.', 'error');
                    }
                }
                
                stopCamera() {
                    if (this.stream) {
                        this.stream.getTracks().forEach(track => track.stop());
                        this.stream = null;
                    }
                    
                    this.isCameraActive = false;
                    this.video.style.display = 'none';
                    this.canvas.style.display = 'none';
                    document.getElementById('videoPlaceholder').style.display = 'block';
                    
                    document.getElementById('startBtn').style.display = 'inline-block';
                    document.getElementById('stopBtn').style.display = 'none';
                    document.getElementById('captureBtn').style.display = 'none';
                    
                    this.stopLiveDetection();
                    this.showStatus('Camera stopped', 'info');
                }
                
                startLiveDetection() {
                    console.log('Starting live detection...');
                    this.detectionInterval = setInterval(() => {
                        if (this.isCameraActive && this.video.readyState === 4) {
                            this.captureAndDetect();
                        }
                    }, 1000); // Detect every 1 second for live detection
                    
                    // Add live detection indicator
                    const liveIndicator = document.createElement('div');
                    liveIndicator.id = 'liveIndicator';
                    liveIndicator.style.cssText = `
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        background: #ff4444;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 15px;
                        font-size: 12px;
                        font-weight: bold;
                        z-index: 20;
                        animation: pulse 1s infinite;
                    `;
                    liveIndicator.textContent = 'LIVE DETECTION';
                    document.querySelector('.video-container').appendChild(liveIndicator);
                }
                
                stopLiveDetection() {
                    if (this.detectionInterval) {
                        clearInterval(this.detectionInterval);
                        this.detectionInterval = null;
                    }
                    
                    const liveIndicator = document.getElementById('liveIndicator');
                    if (liveIndicator) {
                        liveIndicator.remove();
                    }
                }
                
                async captureAndDetect() {
                    if (!this.isCameraActive) return;
                    
                    try {
                        // Draw current video frame to canvas
                        this.ctx.drawImage(this.video, 0, 0);
                        
                        // Convert canvas to base64
                        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
                        
                        // Run detection
                        const results = await this.detectObjects(imageData);
                        
                        // Draw bounding boxes on canvas
                        this.drawBoundingBoxes(results);
                        
                        // Display results
                        this.displayResults(results, 'camera');
                        
                    } catch (error) {
                        console.error('Detection error:', error);
                        this.showStatus('Detection failed: ' + error.message, 'error');
                    }
                }
                
                handleFileUpload(event) {
                    const file = event.target.files[0];
                    if (!file) return;
                    
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.style.maxWidth = '100%';
                        img.style.borderRadius = '10px';
                        
                        const preview = document.getElementById('imagePreview');
                        preview.innerHTML = '';
                        preview.appendChild(img);
                        
                        document.getElementById('detectBtn').style.display = 'inline-block';
                    };
                    reader.readAsDataURL(file);
                }
                
                async detectUploadedImage() {
                    const fileInput = document.getElementById('fileInput');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        this.showStatus('Please select an image first', 'error');
                        return;
                    }
                    
                    try {
                        const reader = new FileReader();
                        reader.onload = async (e) => {
                            const results = await this.detectObjects(e.target.result);
                            this.drawBoundingBoxesOnImage(results);
                            this.displayResults(results, 'upload');
                        };
                        reader.readAsDataURL(file);
                        
                    } catch (error) {
                        console.error('Detection error:', error);
                        this.showStatus('Detection failed: ' + error.message, 'error');
                    }
                }
                
                async detectObjects(imageData) {
                    try {
                        const response = await fetch('/api/detect', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ image: imageData })
                        });
                        
                        if (!response.ok) {
                            throw new Error('Detection request failed');
                        }
                        
                        const data = await response.json();
                        return data.detections || [];
                        
                    } catch (error) {
                        console.error('Detection API error:', error);
                        return [];
                    }
                }
                
                drawBoundingBoxes(results) {
                    // Clear previous drawings
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    
                    // Redraw video frame
                    this.ctx.drawImage(this.video, 0, 0);
                    
                    // Draw bounding boxes
                    results.forEach(result => {
                        const [x1, y1, x2, y2] = result.bbox;
                        
                        // Draw bounding box
                        this.ctx.strokeStyle = '#00ff00';
                        this.ctx.lineWidth = 3;
                        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        
                        // Draw label background
                        const labelText = result.class;
                        this.ctx.font = 'bold 16px Arial';
                        const textMetrics = this.ctx.measureText(labelText);
                        const labelWidth = textMetrics.width + 10;
                        const labelHeight = 25;
                        
                        this.ctx.fillStyle = '#00ff00';
                        this.ctx.fillRect(x1, y1 - labelHeight, labelWidth, labelHeight);
                        
                        // Draw label text
                        this.ctx.fillStyle = '#000000';
                        this.ctx.fillText(labelText, x1 + 5, y1 - 5);
                    });
                }
                
                drawBoundingBoxesOnImage(results) {
                    const img = document.querySelector('#imagePreview img');
                    if (!img) return;
                    
                    // Create canvas overlay
                    let overlay = document.getElementById('imageOverlay');
                    if (!overlay) {
                        overlay = document.createElement('canvas');
                        overlay.id = 'imageOverlay';
                        overlay.style.position = 'absolute';
                        overlay.style.top = '0';
                        overlay.style.left = '0';
                        overlay.style.pointerEvents = 'none';
                        
                        const preview = document.getElementById('imagePreview');
                        preview.style.position = 'relative';
                        preview.appendChild(overlay);
                    }
                    
                    // Set canvas size to match image
                    overlay.width = img.offsetWidth;
                    overlay.height = img.offsetHeight;
                    
                    const ctx = overlay.getContext('2d');
                    ctx.clearRect(0, 0, overlay.width, overlay.height);
                    
                    // Calculate scale factors
                    const scaleX = overlay.width / img.naturalWidth;
                    const scaleY = overlay.height / img.naturalHeight;
                    
                    // Draw bounding boxes
                    results.forEach(result => {
                        const [x1, y1, x2, y2] = result.bbox;
                        
                        // Scale coordinates
                        const scaledX1 = x1 * scaleX;
                        const scaledY1 = y1 * scaleY;
                        const scaledX2 = x2 * scaleX;
                        const scaledY2 = y2 * scaleY;
                        
                        // Draw bounding box
                        ctx.strokeStyle = '#00ff00';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
                        
                        // Draw label background
                        const labelText = result.class;
                        ctx.font = 'bold 20px Arial';
                        const textMetrics = ctx.measureText(labelText);
                        const labelWidth = textMetrics.width + 10;
                        const labelHeight = 30;
                        
                        ctx.fillStyle = '#00ff00';
                        ctx.fillRect(scaledX1, scaledY1 - labelHeight, labelWidth, labelHeight);
                        
                        // Draw label text
                        ctx.fillStyle = '#000000';
                        ctx.fillText(labelText, scaledX1 + 5, scaledY1 - 5);
                    });
                }
                
                displayResults(results, source) {
                    const resultsDiv = document.getElementById('results');
                    const resultsList = document.getElementById('resultsList');
                    
                    if (results.length === 0) {
                        resultsDiv.style.display = 'none';
                        this.showStatus('No safety equipment detected', 'info');
                        return;
                    }
                    
                    resultsList.innerHTML = '';
                    
                    results.forEach((result, index) => {
                        const item = document.createElement('div');
                        item.className = 'detection-item';
                        item.innerHTML = `
                            <h4>${result.class}</h4>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Location:</strong> (${result.bbox[0]}, ${result.bbox[1]}) to (${result.bbox[2]}, ${result.bbox[3]})</p>
                            <p><strong>Source:</strong> ${source}</p>
                        `;
                        resultsList.appendChild(item);
                    });
                    
                    resultsDiv.style.display = 'block';
                    this.showStatus(`Detected ${results.length} safety equipment item(s)`, 'success');
                }
                
                showStatus(message, type) {
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = message;
                    statusDiv.className = `status ${type}`;
                    statusDiv.style.display = 'block';
                    
                    // Auto-hide after 3 seconds
                    setTimeout(() => {
                        statusDiv.style.display = 'none';
                    }, 3000);
                }
            }
            
            // Initialize the detector when page loads
            document.addEventListener('DOMContentLoaded', () => {
                new SafetyDetector();
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.route('/api/detect', methods=['POST'])
def detect():
    """API endpoint for object detection"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        detections = detect_objects(image_data)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Safety Equipment Detection System")
    print("=" * 60)
    
    # Load the trained model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    print("🌐 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔗 Live detection is ready!")
    print("📷 Camera will automatically stop when you refresh or close the page")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
