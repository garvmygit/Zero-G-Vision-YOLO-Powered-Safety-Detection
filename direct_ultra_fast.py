"""
DIRECT ULTRA-FAST TRAINING - NO FILE COPYING
============================================
This script trains directly on the large dataset without copying files
"""

import os
import time
from ultralytics import YOLO
import yaml

def create_direct_config():
    """Create configuration that points directly to the large dataset"""
    
    print("🎯 Creating direct dataset configuration...")
    
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
    
    with open('yolo_params_direct.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Direct dataset configuration created")

def ultra_fast_direct_training():
    """Ultra-fast training directly on large dataset"""
    
    print("🚀 ULTRA-FAST DIRECT TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 30 minutes")
    print("📊 Dataset: 2,107 training + 338 validation images")
    print("⚡ Strategy: Maximum speed optimizations")
    
    start_time = time.time()
    
    # Use YOLOv8n for maximum speed
    model = YOLO("yolo11n.pt")
    
    print("⚡ Training Configuration:")
    print("  - Model: YOLOv8n (nano)")
    print("  - Epochs: 20")
    print("  - Batch Size: 64")
    print("  - Image Size: 256px")
    print("  - Learning Rate: 0.03")
    print("  - Maximum Augmentation")
    
    # Ultra-fast training with maximum optimizations
    results = model.train(
        data="yolo_params_direct.yaml",
        
        # MAXIMUM SPEED SETTINGS
        epochs=20,                    # Reduced epochs
        batch=64,                     # Large batch size
        imgsz=256,                    # Small image size
        device='cpu',                 # CPU training
        
        # AGGRESSIVE LEARNING PARAMETERS
        lr0=0.03,                     # High learning rate
        lrf=0.01,                     # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # MAXIMUM AUGMENTATION FOR ACCURACY
        mosaic=1.0,                   # Maximum mosaic
        mixup=0.3,                    # High mixup
        copy_paste=0.3,               # High copy-paste
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
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=1,             # Minimal warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=10,                 # Early stopping
        save_period=3,               # Frequent saves
        val=True,
        plots=False,                 # Disable plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='ultra_fast_direct_90',
        exist_ok=True,
        
        # ADDITIONAL SPEED OPTIMIZATIONS
        workers=16,                  # Maximum workers
        cache=True,                  # Cache dataset
        amp=True,                    # Mixed precision
        fraction=0.7,                # Use 70% of data for speed
        profile=False,               # Disable profiling
        freeze=None,                 # Don't freeze layers
        multi_scale=False,           # Disable multi-scale
        overlap_mask=True,           # Enable overlap mask
        mask_ratio=4,                # Mask ratio
        dropout=0.0,                 # No dropout
        
        # SPEED HACKS
        close_mosaic=15,             # Close mosaic early
        resume=False,                # Don't resume
        seed=42,                     # Fixed seed
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🎉 ULTRA-FAST TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_direct.yaml", imgsz=256, batch=64)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    success = val_results.box.map50 >= 0.90 and training_time < 30
    print(f"\n🎯 SUCCESS: {'YES' if success else 'NO'}")
    print(f"⏱️  Time target: {'ACHIEVED' if training_time < 30 else 'FAILED'}")
    print(f"🎯 Accuracy target: {'ACHIEVED' if val_results.box.map50 >= 0.90 else 'FAILED'}")
    
    if success:
        print("\n🎉 MISSION ACCOMPLISHED!")
        print("✅ Achieved 90+ accuracy in under 30 minutes!")
    else:
        print("\n⚠️  Mission partially completed:")
        if training_time < 30:
            print("✅ Time target achieved")
        if val_results.box.map50 >= 0.90:
            print("✅ Accuracy target achieved")
    
    return results, training_time, val_results.box.map50

def main():
    """Main ultra-fast direct training workflow"""
    
    print("🚀 ULTRA-FAST DIRECT TRAINING")
    print("=" * 60)
    print("🎯 Goal: 90+ accuracy in under 30 minutes")
    print("📊 Dataset: Large dataset (2,107 + 338 images)")
    print("⚡ Strategy: Direct training with maximum optimizations")
    
    # Step 1: Create direct dataset configuration
    create_direct_config()
    
    # Step 2: Ultra-fast direct training
    results, training_time, accuracy = ultra_fast_direct_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    
    if accuracy >= 0.90 and training_time < 30:
        print("\n🏆 COMPLETE SUCCESS!")
        print("✅ Achieved both time and accuracy targets!")
    elif accuracy >= 0.90:
        print("\n🎯 ACCURACY SUCCESS!")
        print("✅ Achieved 90+ accuracy!")
    elif training_time < 30:
        print("\n⚡ SPEED SUCCESS!")
        print("✅ Completed training in under 30 minutes!")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print("⚠️  Consider running longer or adjusting parameters")

if __name__ == "__main__":
    main()
