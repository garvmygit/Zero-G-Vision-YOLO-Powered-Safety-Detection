"""
QUICK ACCURACY BOOST - Immediate Improvements
============================================
This script makes the most critical improvements for immediate accuracy gains
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from ultralytics import YOLO

def quick_accuracy_fix():
    """Quick fixes for immediate accuracy improvement"""
    
    print("⚡ QUICK ACCURACY BOOST")
    print("=" * 30)
    print("🎯 Making critical improvements...")
    
    # 1. Create more training data quickly
    create_more_data()
    
    # 2. Train with better settings
    train_better_model()
    
    print("✅ Quick accuracy boost completed!")

def create_more_data():
    """Create more training data quickly"""
    
    print("📊 Creating more training data...")
    
    # Create 100 samples per split (vs 10 before)
    samples_per_split = 100
    
    for split in ['train', 'val', 'test']:
        print(f"  Creating {samples_per_split} samples for {split}...")
        
        for i in range(samples_per_split):
            # Create simple but effective images
            img = create_simple_background()
            
            # Add 1-3 objects per image
            num_objects = random.randint(1, 3)
            label_lines = []
            
            for obj_idx in range(num_objects):
                # Create simple colored rectangles as objects
                x1 = random.randint(50, 500)
                y1 = random.randint(50, 500)
                x2 = x1 + random.randint(50, 100)
                y2 = y1 + random.randint(50, 100)
                
                # Draw colored rectangle
                color = random.choice([(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 0, 128)])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                
                # Convert to YOLO format
                center_x = (x1 + x2) / 2 / 640
                center_y = (y1 + y2) / 2 / 640
                width = (x2 - x1) / 640
                height = (y2 - y1) / 640
                
                # Random class
                class_id = random.randint(0, 6)
                
                label_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Save image
            img_file = Path(f"{split}/images") / f"quick_{i:03d}.jpg"
            cv2.imwrite(str(img_file), img)
            
            # Save labels
            label_file = Path(f"{split}/labels") / f"quick_{i:03d}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))
    
    print(f"✅ Created {samples_per_split * 3} additional samples")

def create_simple_background():
    """Create simple background"""
    img = np.random.randint(50, 150, (640, 640, 3), dtype=np.uint8)
    
    # Add some structure
    cv2.rectangle(img, (0, 500), (640, 640), (80, 80, 80), -1)  # Floor
    cv2.rectangle(img, (0, 0), (640, 400), (60, 60, 60), -1)    # Wall
    
    return img

def train_better_model():
    """Train with better settings"""
    
    print("🚀 Training with optimized settings...")
    
    # Use YOLOv8m for better accuracy
    model = YOLO("yolov8m.pt")
    
    results = model.train(
        data="yolo_params.yaml",
        epochs=50,                    # More epochs
        batch=8,                      # Smaller batch
        imgsz=640,
        device='cpu',
        
        # Better hyperparameters
        lr0=0.01,                     # Higher learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Enhanced augmentation
        mosaic=1.0,                   # Maximum mosaic
        mixup=0.1,                    # Mixup
        copy_paste=0.1,               # Copy-paste
        degrees=15.0,                 # Rotation
        translate=0.1,                # Translation
        scale=0.5,                    # Scaling
        
        # Color augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Training settings
        optimizer='AdamW',
        cos_lr=True,
        warmup_epochs=3,
        patience=30,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        
        project='runs/train',
        name='quick_accuracy_boost',
        exist_ok=True
    )
    
    print(f"✅ Training completed! Results: {results.save_dir}")

if __name__ == "__main__":
    quick_accuracy_fix()

