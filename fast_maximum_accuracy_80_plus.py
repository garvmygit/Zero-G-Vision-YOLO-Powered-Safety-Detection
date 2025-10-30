"""
FAST MAXIMUM ACCURACY SOLUTION - 80+ ACCURACY IN UNDER 30 MINUTES
================================================================
This solution uses ALL training data with YOLO11m for 80+ accuracy
- Uses all 1,769 training images
- Uses all 338 validation images  
- Uses YOLO11m (larger model)
- Optimized for SPEED (under 30 minutes)
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
import torch

def create_fast_max_accuracy_config():
    """Create configuration using ALL training data"""
    
    print("🎯 Creating FAST MAXIMUM ACCURACY configuration...")
    print("📊 Using ALL training data for maximum accuracy!")
    print("⚡ Optimized for SPEED (under 30 minutes)")
    
    # Create fast maximum accuracy dataset directories
    for split in ['train', 'val']:
        Path(f"fast_max_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"fast_max_{split}/labels").mkdir(parents=True, exist_ok=True)
    
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
            img_dest = Path(f"fast_max_{split_name}/images") / f"fast_{i:04d}.png"
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"fast_max_{split_name}/labels") / f"fast_{i:04d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_all_samples(train_images, "train")
    copy_all_samples(val_images, "val")
    
    print(f"✅ Created FAST MAXIMUM ACCURACY dataset: {train_samples} train + {val_samples} val samples")
    
    # Create config
    config = {
        'path': '.',
        'train': 'fast_max_train/images',
        'val': 'fast_max_val/images',
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
    
    with open('yolo_params_fast_max.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def fast_maximum_accuracy_training():
    """Fast maximum accuracy training with ALL data and YOLO11m"""
    
    print("🚀 FAST MAXIMUM ACCURACY TRAINING - 80+ ACCURACY IN UNDER 30 MINUTES")
    print("=" * 80)
    print("🎯 Target: 80+ accuracy with ALL data in under 30 minutes")
    print("⚡ Using your RTX 4060 GPU!")
    print("📊 Dataset: ALL DATA (1,769 train + 338 val)")
    print("🤖 Model: YOLO11m (larger model)")
    print("💰 Cost: COMPLETELY FREE!")
    print("⚡ SPEED OPTIMIZATIONS:")
    print("  - 25 epochs (optimized for speed)")
    print("  - Batch size 32 (larger for speed)")
    print("  - Image size 512px (balanced)")
    print("  - Higher learning rate for speed")
    print("  - Minimal augmentation for speed")
    print("  - YOLO11m model (more parameters)")
    print("  - Smart early stopping")
    
    start_time = time.time()
    
    # Use YOLO11m for maximum accuracy (larger model)
    model = YOLO("yolo11m.pt")
    
    # FAST MAXIMUM ACCURACY configuration
    results = model.train(
        data="yolo_params_fast_max.yaml",
        
        # FAST MAXIMUM ACCURACY SETTINGS
        epochs=25,                    # Fewer epochs for speed
        batch=32,                     # Larger batch size for speed
        imgsz=512,                    # Balanced image size
        device='cuda',                # GPU training
        
        # SPEED-OPTIMIZED LEARNING PARAMETERS
        lr0=0.02,                    # Higher learning rate for speed
        lrf=0.002,                   # Higher final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # MINIMAL AUGMENTATION FOR SPEED
        mosaic=0.5,                  # Lower mosaic for speed
        mixup=0.05,                   # Minimal mixup
        copy_paste=0.05,              # Minimal copy-paste
        degrees=5.0,                 # Minimal rotation
        translate=0.05,               # Minimal translation
        scale=0.3,                    # Minimal scaling
        shear=1.0,                    # Minimal shear
        perspective=0.00005,          # Minimal perspective
        
        # MINIMAL COLOR AUGMENTATION
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        
        # OPTIMIZED LOSS WEIGHTS
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # SPEED-OPTIMIZED TRAINING
        optimizer='AdamW',
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=2,             # Minimal warmup for speed
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=8,                  # Shorter patience for speed
        save_period=5,               # Save every 5 epochs
        val=True,                    # Validation
        plots=False,                 # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='fast_max_accuracy_80_plus',
        exist_ok=True,
        
        # GPU OPTIMIZATIONS FOR SPEED
        workers=16,                  # More workers for speed
        cache=True,                  # Cache dataset
        amp=True,                    # Mixed precision for GPU
        fraction=1.0,                # Use all GPU data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # SPEED OPTIMIZATIONS
        close_mosaic=20,             # Close mosaic earlier
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🚀 FAST MAXIMUM ACCURACY TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Final validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_fast_max.yaml", imgsz=512, batch=32)
    
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
    time_success = training_time <= 30
    accuracy_success = val_results.box.map50 >= 0.80
    
    print(f"\n🎯 TARGET ASSESSMENT:")
    print(f"⏱️  Time target (≤30 min): {'ACHIEVED' if time_success else 'FAILED'}")
    print(f"🎯 Accuracy target (80+): {'ACHIEVED' if accuracy_success else 'FAILED'}")
    
    if accuracy_success and time_success:
        print("\n🏆 PERFECT SUCCESS!")
        print("✅ Achieved 80+ accuracy in under 30 minutes!")
        print("🎯 Mission accomplished!")
    elif accuracy_success:
        print("\n🏆 ACCURACY SUCCESS!")
        print("✅ Achieved 80+ accuracy!")
        print(f"⏱️  Time: {training_time:.1f} minutes (target: ≤30)")
    elif time_success:
        print(f"\n🎯 Current accuracy: {val_results.box.map50:.1%}")
        print("✅ Achieved time target!")
        print("💡 To reach 80+ accuracy:")
        print("   - Train for more epochs (30-40)")
        print("   - Use even larger model (yolo11l.pt)")
        print("   - Ensure dataset quality")
    else:
        print(f"\n🎯 Current accuracy: {val_results.box.map50:.1%}")
        print(f"⏱️  Time: {training_time:.1f} minutes")
        print("💡 This is excellent progress!")
        print("💡 To reach both targets:")
        print("   - Use Google Colab with free GPU")
        print("   - Use cloud computing services")
        print("   - Get a more powerful GPU")
    
    print(f"\n💰 COST: COMPLETELY FREE!")
    print("🎯 This is FAST MAXIMUM ACCURACY training!")
    
    return results, training_time, val_results.box.map50

def main():
    """Main fast maximum accuracy training workflow"""
    
    print("🚀 FAST MAXIMUM ACCURACY ANACONDA GPU SOLUTION")
    print("=" * 80)
    print("🎯 Goal: 80+ accuracy with ALL data in under 30 minutes")
    print("📊 Strategy: FAST MAXIMUM ACCURACY TRAINING")
    print("💰 Cost: COMPLETELY FREE!")
    print("⚡ Using your RTX 4060 GPU!")
    print("🤖 Model: YOLO11m (larger model)")
    
    # Check GPU availability
    print(f"\n🔧 GPU Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("✅ GPU ready for FAST MAXIMUM ACCURACY training!")
    else:
        print("⚠️  No GPU detected! Please check your CUDA installation.")
        return
    
    # Create fast maximum accuracy configuration
    create_fast_max_accuracy_config()
    
    # Run fast maximum accuracy training
    results, training_time, accuracy = fast_maximum_accuracy_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    print(f"💰 Cost: COMPLETELY FREE!")
    
    # Check targets
    time_success = training_time <= 30
    accuracy_success = accuracy >= 0.80
    
    print(f"\n🎯 TARGET ASSESSMENT:")
    print(f"⏱️  Time target (≤30 min): {'ACHIEVED' if time_success else 'FAILED'}")
    print(f"🎯 Accuracy target (80+): {'ACHIEVED' if accuracy_success else 'FAILED'}")
    
    if accuracy_success and time_success:
        print("\n🏆 PERFECT SUCCESS!")
        print("✅ Achieved 80+ accuracy in under 30 minutes!")
        print("🎯 Mission accomplished!")
    elif accuracy_success:
        print("\n🏆 ACCURACY SUCCESS!")
        print("✅ Achieved 80+ accuracy!")
        print(f"⏱️  Time: {training_time:.1f} minutes (target: ≤30)")
    elif time_success:
        print(f"\n🎯 Current accuracy: {accuracy:.1%}")
        print("✅ Achieved time target!")
        print("💡 To reach 80+ accuracy:")
        print("   - Train for more epochs (30-40)")
        print("   - Use even larger model (yolo11l.pt)")
        print("   - Ensure dataset quality")
    else:
        print(f"\n🎯 Current accuracy: {accuracy:.1%}")
        print(f"⏱️  Time: {training_time:.1f} minutes")
        print("💡 This is excellent progress!")
        print("💡 To reach both targets:")
        print("   - Use Google Colab with free GPU")
        print("   - Use cloud computing services")
        print("   - Get a more powerful GPU")

if __name__ == "__main__":
    main()
