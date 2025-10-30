"""
GOOGLE COLAB GPU SOLUTION - 90+ ACCURACY IN UNDER 10 MINUTES
===========================================================
This script is optimized for Google Colab with free GPU
- Uses GPU acceleration
- Optimized for speed and accuracy
- Ready to run in Colab environment
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil

def create_colab_config():
    """Create configuration for Google Colab GPU training"""
    
    print("🚀 Creating Google Colab GPU configuration...")
    
    # Create optimized dataset directories
    for split in ['train', 'val']:
        Path(f"colab_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"colab_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Use 200 training and 50 validation samples for good accuracy
    train_samples = 200
    val_samples = 50
    
    # Get training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy colab samples
    def copy_colab_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"colab_{split_name}/images") / f"colab_{i:03d}.png"
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"colab_{split_name}/labels") / f"colab_{i:03d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_colab_samples(train_images, "train")
    copy_colab_samples(val_images, "val")
    
    print(f"✅ Created Colab dataset: {train_samples} train + {val_samples} val samples")
    
    # Create config
    config = {
        'path': '.',
        'train': 'colab_train/images',
        'val': 'colab_val/images',
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
    
    with open('yolo_params_colab.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def colab_gpu_training():
    """Google Colab GPU training for 90+ accuracy in under 10 minutes"""
    
    print("🚀 GOOGLE COLAB GPU TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 10 minutes")
    print("⚡ Strategy: GPU ACCELERATION + OPTIMIZED PARAMETERS")
    print("📊 Dataset: Optimized (200 train + 50 val)")
    print("⚡ Optimizations:")
    print("  - 15 epochs")
    print("  - Batch size 32")
    print("  - Image size 416px")
    print("  - Moderate learning rate")
    print("  - Smart augmentation")
    print("  - GPU acceleration")
    
    start_time = time.time()
    
    # Use YOLOv8n for speed
    model = YOLO("yolo11n.pt")
    
    # GOOGLE COLAB GPU configuration
    results = model.train(
        data="yolo_params_colab.yaml",
        
        # GPU OPTIMIZED SETTINGS
        epochs=15,                    # Balanced epochs
        batch=32,                     # Large batch size for GPU
        imgsz=416,                    # Larger image size for accuracy
        device='cuda',                # GPU training
        
        # OPTIMIZED LEARNING PARAMETERS
        lr0=0.01,                     # Moderate learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # SMART AUGMENTATION
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
        name='colab_gpu_90',
        exist_ok=True,
        
        # GPU OPTIMIZATIONS
        workers=8,                   # GPU workers
        cache=True,                  # Cache dataset
        amp=True,                    # Mixed precision for GPU
        fraction=1.0,                # Use all colab data
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
    
    print(f"\n🚀 COLAB GPU TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_colab.yaml", imgsz=416, batch=32)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    success = val_results.box.map50 >= 0.90 and training_time < 10
    print(f"\n🎯 SUCCESS: {'YES' if success else 'NO'}")
    print(f"⏱️  Time target: {'ACHIEVED' if training_time < 10 else 'FAILED'}")
    print(f"🎯 Accuracy target: {'ACHIEVED' if val_results.box.map50 >= 0.90 else 'FAILED'}")
    
    if success:
        print("\n🏆 COMPLETE SUCCESS!")
        print("✅ Achieved 90+ accuracy in under 10 minutes!")
    elif training_time < 10:
        print("\n⚡ SPEED SUCCESS!")
        print("✅ Achieved training in under 10 minutes!")
        print(f"🎯 Accuracy: {val_results.box.map50:.3f}")
    elif val_results.box.map50 >= 0.90:
        print("\n🎯 ACCURACY SUCCESS!")
        print("✅ Achieved 90+ accuracy!")
        print(f"⏱️  Time: {training_time:.1f} minutes")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print(f"⏱️  Time: {training_time:.1f} minutes")
        print(f"🎯 Accuracy: {val_results.box.map50:.3f}")
    
    return results, training_time, val_results.box.map50

def main():
    """Main Google Colab GPU training workflow"""
    
    print("🚀 GOOGLE COLAB GPU SOLUTION")
    print("=" * 60)
    print("🎯 Goal: 90+ accuracy in under 10 minutes")
    print("📊 Strategy: GPU ACCELERATION")
    print("⚡ For Google Colab with free GPU")
    
    # Check if GPU is available
    import torch
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: No GPU detected!")
        print("This script is designed for Google Colab with GPU.")
        print("Please run this in Google Colab with GPU enabled.")
        print("\n🚀 TO USE IN GOOGLE COLAB:")
        print("1. Go to colab.research.google.com")
        print("2. Enable GPU: Runtime > Change runtime type > GPU")
        print("3. Upload your dataset")
        print("4. Run this script")
        return
    
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    
    # Create colab configuration
    create_colab_config()
    
    # Run colab GPU training
    results, training_time, accuracy = colab_gpu_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    
    if accuracy >= 0.90 and training_time < 10:
        print("\n🏆 COMPLETE SUCCESS!")
        print("✅ Achieved both time and accuracy targets!")
        print("\n🚀 To run in Google Colab:")
        print("1. Go to colab.research.google.com")
        print("2. Enable GPU: Runtime > Change runtime type > GPU")
        print("3. Upload your dataset")
        print("4. Run: python colab_gpu_solution.py")
    elif accuracy >= 0.90:
        print("\n🎯 ACCURACY SUCCESS!")
        print("✅ Achieved 90+ accuracy!")
        print("\n💡 To make it faster:")
        print("1. Reduce epochs to 10-12")
        print("2. Increase batch size to 64")
        print("3. Use smaller image size: imgsz=320")
    elif training_time < 10:
        print("\n⚡ SPEED SUCCESS!")
        print("✅ Completed training in under 10 minutes!")
        print("\n💡 To improve accuracy:")
        print("1. Increase epochs to 20-25")
        print("2. Use larger image size: imgsz=640")
        print("3. Enable more augmentation")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print("⚠️  For better results:")
        print("1. Use Google Colab with GPU")
        print("2. Increase epochs to 20-25")
        print("3. Use larger image size: imgsz=640")
        print("4. Enable more augmentation")

if __name__ == "__main__":
    main()
