"""
HYPER-SPEED TRAINING - MAXIMUM OPTIMIZATION
===========================================
This script implements the most aggressive optimizations possible:
- Minimal data sampling
- Maximum batch size
- Smallest model
- Optimized preprocessing
- Smart early stopping
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import time
from ultralytics import YOLO
import torch
import yaml
from collections import defaultdict

def hyper_speed_training():
    """Hyper-speed training with maximum optimizations"""
    
    print("⚡ HYPER-SPEED TRAINING")
    print("=" * 40)
    print("🎯 Target: 90+ accuracy in under 20 minutes")
    print("🚀 Maximum optimizations enabled")
    
    start_time = time.time()
    
    # Create minimal dataset config
    config = {
        'path': '.',
        'train': '../../hackathon2_train_3/train_3/train3/images',
        'val': '../../hackathon2_train_3/train_3/val3/images',
        'test': '../../hackathon2_test3/test3/images',
        'nc': 7,
        'names': {
            0: 'OxygenTank', 1: 'NitrogenTank', 2: 'FirstAidBox',
            3: 'FireAlarm', 4: 'SafetySwitchPanel', 5: 'EmergencyPhone',
            6: 'FireExtinguisher'
        }
    }
    
    with open('yolo_params_hyper.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Use YOLOv8n for maximum speed
    model = YOLO("yolo11n.pt")
    
    # HYPER-SPEED configuration
    results = model.train(
        data="yolo_params_hyper.yaml",
        
        # MAXIMUM SPEED SETTINGS
        epochs=20,                    # Minimal epochs
        batch=64,                     # Maximum batch size
        imgsz=256,                    # Smallest practical size
        device='cpu',
        
        # AGGRESSIVE LEARNING
        lr0=0.03,                     # Very high learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # MAXIMUM AUGMENTATION
        mosaic=1.0,
        mixup=0.3,                    # Maximum mixup
        copy_paste=0.3,               # Maximum copy-paste
        degrees=45.0,                 # Maximum rotation
        translate=0.5,                # Maximum translation
        scale=1.0,                    # Maximum scaling
        shear=10.0,                   # Maximum shear
        perspective=0.0005,           # Maximum perspective
        
        # COLOR AUGMENTATION
        hsv_h=0.03,
        hsv_s=1.0,
        hsv_v=0.6,
        
        # LOSS OPTIMIZATION
        box=10.0,                     # Higher box loss weight
        cls=1.0,                      # Higher class loss weight
        dfl=2.0,                      # Higher DFL loss weight
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,
        warmup_epochs=1,             # Minimal warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION
        patience=10,                 # Early stopping
        save_period=3,               # Frequent saves
        val=True,
        plots=False,                 # Disable plots for speed
        verbose=True,
        
        # PROJECT
        project='runs/train',
        name='hyper_speed_90_accuracy',
        exist_ok=True,
        
        # ADDITIONAL OPTIMIZATIONS
        workers=16,                  # Maximum workers
        cache=True,                  # Cache dataset
        amp=True,                    # Mixed precision
        fraction=0.8,                # Use 80% of data for speed
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # SPEED HACKS
        close_mosaic=15,             # Close mosaic early
        resume=False,                # Don't resume
        seed=42,                     # Fixed seed for reproducibility
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⚡ HYPER-SPEED TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_hyper.yaml", imgsz=256, batch=64)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    success = val_results.box.map50 >= 0.90 and training_time < 30
    print(f"🎯 SUCCESS: {'YES' if success else 'NO'}")
    
    return results, training_time, val_results.box.map50

if __name__ == "__main__":
    hyper_speed_training()
