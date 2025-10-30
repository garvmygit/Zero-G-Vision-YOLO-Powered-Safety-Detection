#!/usr/bin/env python3
"""
Quick YOLO Training - 60 epochs, GPU, large batch, large images
Target: 80%+ accuracy in 50 minutes
"""

import os
import sys
import time
import yaml
import shutil
from pathlib import Path

def setup_dataset():
    """Setup dataset from local files"""
    print("Setting up dataset...")
    
    # Define paths
    train_images_path = Path("../hackathon2_train_3/train_3/train3/images")
    val_images_path = Path("../hackathon2_train_3/train_3/val3/images")
    
    # Check if dataset exists
    if not train_images_path.exists():
        print(f"Training dataset not found at: {train_images_path}")
        return False
    
    if not val_images_path.exists():
        print(f"Validation dataset not found at: {val_images_path}")
        return False
    
    # Create directories
    Path("train/images").mkdir(parents=True, exist_ok=True)
    Path("train/labels").mkdir(parents=True, exist_ok=True)
    Path("val/images").mkdir(parents=True, exist_ok=True)
    Path("val/labels").mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    train_images = list(train_images_path.glob("*.png"))
    print(f"Found {len(train_images)} training images")
    
    for i, img_path in enumerate(train_images):
        shutil.copy2(img_path, f"train/images/train_{i:04d}.png")
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, f"train/labels/train_{i:04d}.txt")
    
    # Copy validation images
    val_images = list(val_images_path.glob("*.png"))
    print(f"Found {len(val_images)} validation images")
    
    for i, img_path in enumerate(val_images):
        shutil.copy2(img_path, f"val/images/val_{i:04d}.png")
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, f"val/labels/val_{i:04d}.txt")
    
    print(f"Dataset ready: {len(train_images)} train + {len(val_images)} val")
    
    # Create YOLO config
    config = {
        'path': '.',
        'train': 'train/images',
        'val': 'val/images',
        'nc': 7,
        'names': {
            0: 'OxygenTank',
            1: 'NitrogenTank', 
            2: 'FirstAidBox',
            3: 'FireAlarm',
            4: 'SafetySwitchPanel',
            5: 'EmergencyPhone',
            6: 'FireExtinguisher'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Dataset configuration created")
    return True

def train_model():
    """Train YOLO model"""
    print("Starting YOLO training...")
    print("Target: 80%+ accuracy in 50 minutes")
    print("Settings: 60 epochs, GPU, batch=32, imgsz=640")
    
    try:
        from ultralytics import YOLO
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        start_time = time.time()
        
        # Load model
        model = YOLO("yolo11m.pt")
        
        # Train
        results = model.train(
            data="dataset.yaml",
            epochs=60,
            batch=32,
            imgsz=640,
            device=device,
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.2,
            degrees=15.0,
            translate=0.2,
            scale=0.8,
            shear=5.0,
            perspective=0.0002,
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            patience=15,
            save_period=10,
            val=True,
            plots=False,
            verbose=True,
            project='runs/train',
            name='yolo_80_accuracy',
            exist_ok=True,
            workers=8,
            cache='disk',
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            close_mosaic=50,
            resume=False,
            seed=42,
        )
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed!")
        print(f"Time: {training_time:.1f} minutes")
        print(f"Results: {results.save_dir}")
        
        # Validate
        print("Validating model...")
        model = YOLO(results.save_dir / "weights" / "best.pt")
        val_results = model.val(data="dataset.yaml", imgsz=640, batch=32)
        
        print(f"\nFinal Results:")
        print(f"mAP50: {val_results.box.map50:.3f}")
        print(f"mAP50-95: {val_results.box.map:.3f}")
        print(f"Precision: {val_results.box.mp:.3f}")
        print(f"Recall: {val_results.box.mr:.3f}")
        
        accuracy = val_results.box.map50
        print(f"\nAccuracy: {accuracy:.1%}")
        
        if accuracy >= 0.80:
            print("SUCCESS! Achieved 80%+ accuracy!")
        else:
            print("Close to target! Consider more epochs for 80%+")
        
        print(f"\nTrained model saved to: {results.save_dir}/weights/best.pt")
        
        return results, val_results
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    """Main function"""
    print("YOLO Training - 80% Accuracy Target")
    print("=" * 50)
    
    # Setup dataset
    if not setup_dataset():
        print("Failed to setup dataset")
        return
    
    # Train model
    results, val_results = train_model()
    
    if results:
        print("\nTraining completed successfully!")
        print(f"Model: {results.save_dir}/weights/best.pt")
        if val_results:
            print(f"Accuracy: {val_results.box.map50:.1%}")

if __name__ == "__main__":
    main()
