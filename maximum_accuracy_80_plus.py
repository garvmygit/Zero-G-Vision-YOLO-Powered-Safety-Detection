"""
MAXIMUM ACCURACY SOLUTION - 80+ ACCURACY WITH ALL DATA
====================================================
This solution uses ALL training data with YOLO11m for 80+ accuracy
- Uses all 1,769 training images
- Uses all 338 validation images  
- Uses YOLO11m (larger model)
- Optimized for 50-60 epochs
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
import torch

def create_maximum_accuracy_config():
    """Create configuration using ALL training data"""
    
    print("🎯 Creating MAXIMUM ACCURACY configuration...")
    print("📊 Using ALL training data for maximum accuracy!")
    
    # Create maximum accuracy dataset directories
    for split in ['train', 'val']:
        Path(f"max_accuracy_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"max_accuracy_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Use ALL data for maximum accuracy
    train_samples = 1769  # ALL training images
    val_samples = 338     # ALL validation images
    
    print(f"📊 Dataset: ALL DATA ({train_samples} train + {val_samples} val)")
    
    # Get ALL training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get ALL validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy ALL samples for maximum accuracy
    def copy_all_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"max_accuracy_{split_name}/images") / f"max_{i:04d}.png"
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"max_accuracy_{split_name}/labels") / f"max_{i:04d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_all_samples(train_images, "train")
    copy_all_samples(val_images, "val")
    
    print(f"✅ Created MAXIMUM ACCURACY dataset: {train_samples} train + {val_samples} val samples")
    
    # Create config
    config = {
        'path': '.',
        'train': 'max_accuracy_train/images',
        'val': 'max_accuracy_val/images',
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
    
    with open('yolo_params_max_accuracy.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def maximum_accuracy_training():
    """Maximum accuracy training with ALL data and YOLO11m"""
    
    print("🚀 MAXIMUM ACCURACY TRAINING - 80+ ACCURACY TARGET")
    print("=" * 70)
    print("🎯 Target: 80+ accuracy with ALL data")
    print("⚡ Using your RTX 4060 GPU!")
    print("📊 Dataset: ALL DATA (1,769 train + 338 val)")
    print("🤖 Model: YOLO11m (larger model)")
    print("💰 Cost: COMPLETELY FREE!")
    print("⚡ Optimizations:")
    print("  - 60 epochs (more training)")
    print("  - Batch size 16 (optimal for larger model)")
    print("  - Image size 640px (larger for accuracy)")
    print("  - Smart learning rate scheduling")
    print("  - Optimized augmentation")
    print("  - YOLO11m model (more parameters)")
    print("  - Smart early stopping")
    
    start_time = time.time()
    
    # Use YOLO11m for maximum accuracy (larger model)
    model = YOLO("yolo11m.pt")
    
    # MAXIMUM ACCURACY configuration
    results = model.train(
        data="yolo_params_max_accuracy.yaml",
        
        # MAXIMUM ACCURACY SETTINGS
        epochs=60,                    # More epochs for better accuracy
        batch=16,                     # Optimal batch size for larger model
        imgsz=640,                    # Larger image size for accuracy
        device='cuda',                # GPU training
        
        # OPTIMIZED LEARNING PARAMETERS FOR MAXIMUM ACCURACY
        lr0=0.006,                   # Lower learning rate for stability
        lrf=0.0006,                  # Lower final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # OPTIMIZED AUGMENTATION FOR MAXIMUM ACCURACY
        mosaic=0.95,                  # High mosaic
        mixup=0.12,                   # Moderate mixup
        copy_paste=0.12,              # Moderate copy-paste
        degrees=8.0,                 # Moderate rotation
        translate=0.12,               # Moderate translation
        scale=0.5,                    # Moderate scaling
        shear=2.5,                    # Moderate shear
        perspective=0.00012,          # Moderate perspective
        
        # ENHANCED COLOR AUGMENTATION
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # OPTIMIZED LOSS WEIGHTS
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # SMART TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=6,             # More warmup for larger model
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=15,                 # Smart early stopping
        save_period=10,              # Save every 10 epochs
        val=True,                    # Validation
        plots=False,                 # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='max_accuracy_80_plus',
        exist_ok=True,
        
        # GPU OPTIMIZATIONS
        workers=8,                   # GPU workers
        cache=True,                  # Cache dataset
        amp=True,                    # Mixed precision for GPU
        fraction=1.0,                # Use all GPU data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # TRAINING HACKS
        close_mosaic=50,             # Close mosaic later
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🚀 MAXIMUM ACCURACY TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Final validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_max_accuracy.yaml", imgsz=640, batch=16)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    print(f"\n🎯 ACCURACY ASSESSMENT:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {val_results.box.map50:.3f}")
    print(f"💰 Cost: COMPLETELY FREE!")
    
    # Check targets
    accuracy_success = val_results.box.map50 >= 0.80
    
    print(f"\n🎯 TARGET ASSESSMENT:")
    print(f"🎯 Accuracy target (80+): {'ACHIEVED' if accuracy_success else 'FAILED'}")
    
    if accuracy_success:
        print("\n🏆 SUCCESS!")
        print("✅ Achieved 80+ accuracy!")
        print("🎯 Mission accomplished!")
    else:
        print(f"\n🎯 Current accuracy: {val_results.box.map50:.1%}")
        print("💡 This is excellent progress!")
        print("💡 To reach 80+ accuracy:")
        print("   - Train for more epochs (75-100)")
        print("   - Use even larger model (yolo11l.pt)")
        print("   - Ensure dataset quality")
        print("   - Try different augmentation strategies")
    
    print(f"\n💰 COST: COMPLETELY FREE!")
    print("🎯 This is MAXIMUM ACCURACY training!")
    
    return results, training_time, val_results.box.map50

def main():
    """Main maximum accuracy training workflow"""
    
    print("🚀 MAXIMUM ACCURACY ANACONDA GPU SOLUTION")
    print("=" * 70)
    print("🎯 Goal: 80+ accuracy with ALL data")
    print("📊 Strategy: MAXIMUM ACCURACY TRAINING")
    print("💰 Cost: COMPLETELY FREE!")
    print("⚡ Using your RTX 4060 GPU!")
    print("🤖 Model: YOLO11m (larger model)")
    
    # Check GPU availability
    print(f"\n🔧 GPU Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("✅ GPU ready for MAXIMUM ACCURACY training!")
    else:
        print("⚠️  No GPU detected! Please check your CUDA installation.")
        return
    
    # Create maximum accuracy configuration
    create_maximum_accuracy_config()
    
    # Run maximum accuracy training
    results, training_time, accuracy = maximum_accuracy_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    print(f"💰 Cost: COMPLETELY FREE!")
    
    # Check targets
    accuracy_success = accuracy >= 0.80
    
    print(f"\n🎯 TARGET ASSESSMENT:")
    print(f"🎯 Accuracy target (80+): {'ACHIEVED' if accuracy_success else 'FAILED'}")
    
    if accuracy_success:
        print("\n🏆 MISSION ACCOMPLISHED!")
        print("✅ Achieved 80+ accuracy!")
        print("🎯 Target achieved!")
    else:
        print(f"\n🎯 Current accuracy: {accuracy:.1%}")
        print("💡 This is excellent progress!")
        print("💡 To reach 80+ accuracy:")
        print("   - Train for more epochs (75-100)")
        print("   - Use even larger model (yolo11l.pt)")
        print("   - Ensure dataset quality")
        print("   - Try different augmentation strategies")

if __name__ == "__main__":
    main()
