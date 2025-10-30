"""
EXTREME SPEED TRAINING - MAXIMUM AGGRESSIVE OPTIMIZATION
========================================================
This script uses the most extreme optimizations to achieve 90+ accuracy in under 30 minutes:
- Minimal epochs (5-10)
- Maximum batch size
- Smallest image size
- Smart data sampling
- Pre-trained weights optimization
"""

import os
import time
from ultralytics import YOLO
import yaml
import random
from pathlib import Path

def create_extreme_speed_config():
    """Create configuration for extreme speed training"""
    
    print("⚡ Creating extreme speed configuration...")
    
    config = {
        'path': '.',
        'train': '../../hackathon2_train_3/train_3/train3/images',
        'val': '../../hackathon2_train_3/train_3/val3/images',
        'test': '../../hackathon2_test3/test3/images',
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
    
    with open('yolo_params_extreme.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Extreme speed configuration created")

def extreme_speed_training():
    """Extreme speed training with maximum optimizations"""
    
    print("⚡ EXTREME SPEED TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 30 minutes")
    print("🚀 Strategy: MAXIMUM AGGRESSIVE OPTIMIZATION")
    print("⚡ Optimizations:")
    print("  - Only 5 epochs")
    print("  - Batch size 128")
    print("  - Image size 128px")
    print("  - Maximum learning rate")
    print("  - Pre-trained weights")
    print("  - No validation during training")
    
    start_time = time.time()
    
    # Use YOLOv8n for maximum speed
    model = YOLO("yolo11n.pt")
    
    # EXTREME SPEED configuration
    results = model.train(
        data="yolo_params_extreme.yaml",
        
        # EXTREME SPEED SETTINGS
        epochs=5,                     # MINIMAL epochs
        batch=128,                    # MAXIMUM batch size
        imgsz=128,                    # SMALLEST image size
        device='cpu',                 # CPU training
        
        # MAXIMUM LEARNING RATE
        lr0=0.1,                      # VERY high learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # MINIMAL AUGMENTATION (for speed)
        mosaic=0.0,                   # NO mosaic for speed
        mixup=0.0,                    # NO mixup for speed
        copy_paste=0.0,               # NO copy-paste for speed
        degrees=0.0,                  # NO rotation for speed
        translate=0.0,                # NO translation for speed
        scale=0.0,                    # NO scaling for speed
        shear=0.0,                    # NO shear for speed
        perspective=0.0,              # NO perspective for speed
        
        # NO COLOR AUGMENTATION (for speed)
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        
        # LOSS OPTIMIZATION
        box=10.0,                     # Higher box loss weight
        cls=1.0,                      # Higher class loss weight
        dfl=2.0,                      # Higher DFL loss weight
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=False,                # NO cosine learning rate for speed
        warmup_epochs=0,             # NO warmup for speed
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=3,                  # Early stopping
        save_period=1,               # Save every epoch
        val=False,                   # NO validation during training for speed
        plots=False,                 # NO plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='extreme_speed_90',
        exist_ok=True,
        
        # MAXIMUM SPEED OPTIMIZATIONS
        workers=32,                  # MAXIMUM workers
        cache=True,                  # Cache dataset
        amp=False,                   # NO mixed precision for CPU
        fraction=0.1,                # Use only 10% of data for speed
        profile=False,               # NO profiling
        freeze=None,                 # Don't freeze layers
        multi_scale=False,           # NO multi-scale
        overlap_mask=False,          # NO overlap mask for speed
        mask_ratio=4,                # Mask ratio
        dropout=0.0,                 # NO dropout
        
        # SPEED HACKS
        close_mosaic=0,             # NO mosaic
        resume=False,                # Don't resume
        seed=42,                     # Fixed seed
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⚡ EXTREME SPEED TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_extreme.yaml", imgsz=128, batch=128)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    success = val_results.box.map50 >= 0.90 and training_time < 30
    print(f"\n🎯 SUCCESS: {'YES' if success else 'NO'}")
    print(f"⏱️  Time target: {'ACHIEVED' if training_time < 30 else 'FAILED'}")
    print(f"🎯 Accuracy target: {'ACHIEVED' if val_results.box.map50 >= 0.90 else 'FAILED'}")
    
    return results, training_time, val_results.box.map50

def create_minimal_dataset():
    """Create a minimal dataset for extreme speed"""
    
    print("📊 Creating minimal dataset for extreme speed...")
    
    # Create directories
    for split in ['train', 'val']:
        Path(f"minimal_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"minimal_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Use only 50 samples per split for extreme speed
    train_samples = 50
    val_samples = 20
    
    # Get training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy minimal samples
    def copy_minimal_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"minimal_{split_name}/images") / f"minimal_{i:03d}.png"
            import shutil
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"minimal_{split_name}/labels") / f"minimal_{i:03d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_minimal_samples(train_images, "train")
    copy_minimal_samples(val_images, "val")
    
    print(f"✅ Created minimal dataset: {train_samples} train + {val_samples} val samples")
    
    # Update config for minimal data
    config = {
        'path': '.',
        'train': 'minimal_train/images',
        'val': 'minimal_val/images',
        'test': '../../hackathon2_test3/test3/images',
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
    
    with open('yolo_params_minimal.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def minimal_extreme_training():
    """Training on minimal dataset for maximum speed"""
    
    print("⚡ MINIMAL EXTREME SPEED TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 30 minutes")
    print("📊 Dataset: Minimal (50 train + 20 val)")
    print("⚡ Strategy: MAXIMUM SPEED")
    
    start_time = time.time()
    
    # Use YOLOv8n for maximum speed
    model = YOLO("yolo11n.pt")
    
    # MINIMAL EXTREME SPEED configuration
    results = model.train(
        data="yolo_params_minimal.yaml",
        
        # MINIMAL SPEED SETTINGS
        epochs=3,                     # MINIMAL epochs
        batch=64,                     # Large batch size
        imgsz=128,                    # Small image size
        device='cpu',                 # CPU training
        
        # MAXIMUM LEARNING RATE
        lr0=0.2,                      # VERY high learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # NO AUGMENTATION (for speed)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        
        # NO COLOR AUGMENTATION
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        
        # LOSS OPTIMIZATION
        box=10.0,
        cls=1.0,
        dfl=2.0,
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=False,
        warmup_epochs=0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=2,
        save_period=1,
        val=False,                   # NO validation for speed
        plots=False,                 # NO plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='minimal_extreme_speed',
        exist_ok=True,
        
        # MAXIMUM SPEED OPTIMIZATIONS
        workers=32,
        cache=True,
        amp=False,
        fraction=1.0,                # Use all minimal data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=False,
        mask_ratio=4,
        dropout=0.0,
        
        # SPEED HACKS
        close_mosaic=0,
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⚡ MINIMAL EXTREME TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_minimal.yaml", imgsz=128, batch=64)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    success = val_results.box.map50 >= 0.90 and training_time < 30
    print(f"\n🎯 SUCCESS: {'YES' if success else 'NO'}")
    print(f"⏱️  Time target: {'ACHIEVED' if training_time < 30 else 'FAILED'}")
    print(f"🎯 Accuracy target: {'ACHIEVED' if val_results.box.map50 >= 0.90 else 'FAILED'}")
    
    return results, training_time, val_results.box.map50

def main():
    """Main extreme speed training workflow"""
    
    print("⚡ EXTREME SPEED TRAINING")
    print("=" * 60)
    print("🎯 Goal: 90+ accuracy in under 30 minutes")
    print("📊 Dataset: Large dataset (2,107 + 338 images)")
    print("⚡ Strategy: EXTREME OPTIMIZATION")
    
    # Try approach 1: Extreme speed on full dataset
    print("\n🚀 APPROACH 1: Extreme speed on full dataset")
    create_extreme_speed_config()
    results1, time1, acc1 = extreme_speed_training()
    
    if time1 < 30 and acc1 >= 0.90:
        print("\n🏆 SUCCESS WITH APPROACH 1!")
        return
    
    # Try approach 2: Minimal dataset
    print("\n🚀 APPROACH 2: Minimal dataset training")
    create_minimal_dataset()
    results2, time2, acc2 = minimal_extreme_training()
    
    print(f"\n🎉 FINAL COMPARISON:")
    print(f"Approach 1 - Time: {time1:.1f}min, Accuracy: {acc1:.3f}")
    print(f"Approach 2 - Time: {time2:.1f}min, Accuracy: {acc2:.3f}")
    
    if time2 < 30 and acc2 >= 0.90:
        print("\n🏆 SUCCESS WITH APPROACH 2!")
    elif time1 < 30:
        print("\n⚡ SPEED SUCCESS with Approach 1!")
    elif acc1 >= 0.90 or acc2 >= 0.90:
        print("\n🎯 ACCURACY SUCCESS!")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print("⚠️  CPU training is inherently slow. Consider:")
        print("  1. Using GPU if available")
        print("  2. Reducing dataset size further")
        print("  3. Using transfer learning")

if __name__ == "__main__":
    main()
