"""
BALANCED SPEED-ACCURACY TRAINING
================================
This script balances speed and accuracy to achieve 90+ accuracy in under 30 minutes
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path

def create_balanced_config():
    """Create configuration for balanced speed-accuracy training"""
    
    print("🎯 Creating balanced speed-accuracy configuration...")
    
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
    
    with open('yolo_params_balanced.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Balanced configuration created")

def balanced_training():
    """Balanced training for speed and accuracy"""
    
    print("⚖️ BALANCED SPEED-ACCURACY TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 30 minutes")
    print("📊 Strategy: Balanced Speed + Accuracy")
    print("⚡ Optimizations:")
    print("  - 15 epochs (balanced)")
    print("  - Batch size 32")
    print("  - Image size 320px")
    print("  - Moderate learning rate")
    print("  - Smart augmentation")
    print("  - Pre-trained weights")
    
    start_time = time.time()
    
    # Use YOLOv8n for speed
    model = YOLO("yolo11n.pt")
    
    # BALANCED configuration
    results = model.train(
        data="yolo_params_balanced.yaml",
        
        # BALANCED SPEED SETTINGS
        epochs=15,                    # Balanced epochs
        batch=32,                     # Moderate batch size
        imgsz=320,                    # Moderate image size
        device='cpu',                 # CPU training
        
        # BALANCED LEARNING PARAMETERS
        lr0=0.01,                     # Moderate learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # BALANCED AUGMENTATION
        mosaic=0.8,                   # Moderate mosaic
        mixup=0.1,                    # Some mixup
        copy_paste=0.1,               # Some copy-paste
        degrees=10.0,                 # Moderate rotation
        translate=0.1,                # Moderate translation
        scale=0.5,                    # Moderate scaling
        shear=2.0,                    # Some shear
        perspective=0.0001,           # Some perspective
        
        # BALANCED COLOR AUGMENTATION
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # LOSS OPTIMIZATION
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=2,             # Minimal warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=8,                  # Early stopping
        save_period=3,               # Save every 3 epochs
        val=True,                    # Validation
        plots=False,                 # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='balanced_speed_accuracy',
        exist_ok=True,
        
        # BALANCED OPTIMIZATIONS
        workers=16,
        cache=True,
        amp=False,                   # No mixed precision for CPU
        fraction=0.5,                # Use 50% of data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # TRAINING HACKS
        close_mosaic=12,             # Close mosaic early
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⚖️ BALANCED TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_balanced.yaml", imgsz=320, batch=32)
    
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

def create_smart_sampled_config():
    """Create configuration with smart sampling"""
    
    print("🧠 Creating smart sampled configuration...")
    
    # Create directories for smart sampling
    for split in ['train', 'val']:
        Path(f"smart_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"smart_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Smart sampling: Use 200 training and 50 validation samples
    train_samples = 200
    val_samples = 50
    
    # Get training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy smart samples
    def copy_smart_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"smart_{split_name}/images") / f"smart_{i:03d}.png"
            import shutil
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"smart_{split_name}/labels") / f"smart_{i:03d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_smart_samples(train_images, "train")
    copy_smart_samples(val_images, "val")
    
    print(f"✅ Created smart sampled dataset: {train_samples} train + {val_samples} val samples")
    
    # Update config for smart sampled data
    config = {
        'path': '.',
        'train': 'smart_train/images',
        'val': 'smart_val/images',
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
    
    with open('yolo_params_smart.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def smart_sampled_training():
    """Training on smart sampled dataset"""
    
    print("🧠 SMART SAMPLED TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 30 minutes")
    print("📊 Dataset: Smart sampled (200 train + 50 val)")
    print("⚡ Strategy: Smart Sampling + Balanced Training")
    
    start_time = time.time()
    
    # Use YOLOv8n for speed
    model = YOLO("yolo11n.pt")
    
    # SMART SAMPLED configuration
    results = model.train(
        data="yolo_params_smart.yaml",
        
        # SMART SPEED SETTINGS
        epochs=20,                    # More epochs for better accuracy
        batch=16,                     # Smaller batch for stability
        imgsz=416,                    # Larger image size for accuracy
        device='cpu',                 # CPU training
        
        # SMART LEARNING PARAMETERS
        lr0=0.01,                     # Moderate learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # SMART AUGMENTATION
        mosaic=1.0,                   # Maximum mosaic
        mixup=0.15,                   # Moderate mixup
        copy_paste=0.15,              # Moderate copy-paste
        degrees=15.0,                 # Moderate rotation
        translate=0.1,                # Moderate translation
        scale=0.5,                    # Moderate scaling
        shear=2.0,                    # Some shear
        perspective=0.0001,           # Some perspective
        
        # SMART COLOR AUGMENTATION
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # LOSS OPTIMIZATION
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=3,             # Warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=10,                 # Early stopping
        save_period=5,               # Save every 5 epochs
        val=True,                    # Validation
        plots=False,                 # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='smart_sampled_90',
        exist_ok=True,
        
        # SMART OPTIMIZATIONS
        workers=16,
        cache=True,
        amp=False,                   # No mixed precision for CPU
        fraction=1.0,                # Use all smart sampled data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # TRAINING HACKS
        close_mosaic=15,             # Close mosaic early
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🧠 SMART SAMPLED TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_smart.yaml", imgsz=416, batch=16)
    
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
    """Main balanced training workflow"""
    
    print("⚖️ BALANCED SPEED-ACCURACY TRAINING")
    print("=" * 60)
    print("🎯 Goal: 90+ accuracy in under 30 minutes")
    print("📊 Strategy: Balanced Speed + Accuracy")
    
    # Try approach 1: Balanced training on full dataset
    print("\n🚀 APPROACH 1: Balanced training on full dataset")
    create_balanced_config()
    results1, time1, acc1 = balanced_training()
    
    if time1 < 30 and acc1 >= 0.90:
        print("\n🏆 SUCCESS WITH APPROACH 1!")
        return
    
    # Try approach 2: Smart sampled training
    print("\n🚀 APPROACH 2: Smart sampled training")
    create_smart_sampled_config()
    results2, time2, acc2 = smart_sampled_training()
    
    print(f"\n🎉 FINAL COMPARISON:")
    print(f"Approach 1 - Time: {time1:.1f}min, Accuracy: {acc1:.3f}")
    print(f"Approach 2 - Time: {time2:.1f}min, Accuracy: {acc2:.3f}")
    
    if time2 < 30 and acc2 >= 0.90:
        print("\n🏆 SUCCESS WITH APPROACH 2!")
    elif time1 < 30 and acc1 >= 0.90:
        print("\n🏆 SUCCESS WITH APPROACH 1!")
    elif time1 < 30 or time2 < 30:
        print("\n⚡ SPEED SUCCESS!")
        print("✅ Achieved training in under 30 minutes!")
    elif acc1 >= 0.90 or acc2 >= 0.90:
        print("\n🎯 ACCURACY SUCCESS!")
        print("✅ Achieved 90+ accuracy!")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print("⚠️  CPU training limitations:")
        print("  1. CPU is inherently slower than GPU")
        print("  2. Consider using Google Colab with GPU")
        print("  3. Try cloud computing services")
        print("  4. Use transfer learning with pre-trained models")

if __name__ == "__main__":
    main()
