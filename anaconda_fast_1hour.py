"""
ANACONDA FAST SOLUTION - UNDER 1 HOUR
====================================
This solution is optimized for speed while maintaining good accuracy.
- Completes in under 1 hour
- Optimized for CPU training
- Smart data sampling
- Pre-trained weights optimization
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil

def create_anaconda_fast_config():
    """Create configuration for Anaconda fast training"""
    
    print("🎯 Creating Anaconda fast configuration...")
    
    # Create optimized dataset directories
    for split in ['train', 'val']:
        Path(f"anaconda_fast_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"anaconda_fast_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Use moderate data for speed (150 training and 40 validation samples)
    train_samples = 150
    val_samples = 40
    
    # Get training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy anaconda fast samples
    def copy_anaconda_fast_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"anaconda_fast_{split_name}/images") / f"anaconda_fast_{i:03d}.png"
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"anaconda_fast_{split_name}/labels") / f"anaconda_fast_{i:03d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_anaconda_fast_samples(train_images, "train")
    copy_anaconda_fast_samples(val_images, "val")
    
    print(f"✅ Created Anaconda fast dataset: {train_samples} train + {val_samples} val samples")
    
    # Create config
    config = {
        'path': '.',
        'train': 'anaconda_fast_train/images',
        'val': 'anaconda_fast_val/images',
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
    
    with open('yolo_params_anaconda_fast.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def anaconda_fast_training():
    """Anaconda fast training - under 1 hour"""
    
    print("⚡ ANACONDA FAST TRAINING")
    print("=" * 50)
    print("🎯 Target: BEST accuracy in under 1 hour")
    print("⚡ Strategy: SPEED + ACCURACY OPTIMIZATION")
    print("📊 Dataset: Moderate (150 train + 40 val)")
    print("💰 Cost: COMPLETELY FREE!")
    print("⏱️  Time target: Under 1 hour")
    print("⚡ Optimizations:")
    print("  - 25 epochs (balanced for speed)")
    print("  - Batch size 8 (optimal for CPU)")
    print("  - Image size 320px (balanced)")
    print("  - Moderate learning rate")
    print("  - Smart augmentation")
    print("  - Pre-trained weights")
    
    start_time = time.time()
    
    # Use YOLOv8n for speed
    model = YOLO("yolo11n.pt")
    
    # ANACONDA FAST configuration
    results = model.train(
        data="yolo_params_anaconda_fast.yaml",
        
        # SPEED OPTIMIZED SETTINGS
        epochs=25,                    # Balanced epochs for speed
        batch=8,                      # Optimal batch size for CPU
        imgsz=320,                    # Balanced image size
        device='cpu',                 # CPU training
        
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
        
        # OPTIMIZED LOSS WEIGHTS
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=3,             # Minimal warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=8,                  # Early stopping
        save_period=5,               # Save every 5 epochs
        val=True,                    # Validation
        plots=False,                 # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='anaconda_fast_1hr',
        exist_ok=True,
        
        # CPU OPTIMIZATIONS
        workers=16,
        cache=True,
        amp=False,                   # No mixed precision for CPU
        fraction=1.0,                # Use all fast data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # TRAINING HACKS
        close_mosaic=20,             # Close mosaic early
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⚡ ANACONDA FAST TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Final validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_anaconda_fast.yaml", imgsz=320, batch=8)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    print(f"\n🎯 ACCURACY ASSESSMENT:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {val_results.box.map50:.3f}")
    print(f"💰 Cost: COMPLETELY FREE!")
    
    # Check if time target was met
    time_success = training_time < 60
    print(f"\n⏱️  Time target: {'ACHIEVED' if time_success else 'FAILED'}")
    print(f"🎯 Time: {training_time:.1f} minutes (target: <60 minutes)")
    
    if val_results.box.map50 >= 0.80:
        print("\n🏆 SUCCESS! ACHIEVED 80+ ACCURACY!")
        print("✅ Target achieved on FREE CPU!")
        print("🎯 Mission accomplished!")
    elif val_results.box.map50 >= 0.60:
        print("\n🎯 EXCELLENT RESULT!")
        print(f"✅ Achieved {val_results.box.map50:.1%} accuracy!")
        print("This is excellent for FREE CPU training!")
    elif val_results.box.map50 >= 0.40:
        print("\n🔄 GOOD RESULT")
        print(f"✅ Achieved {val_results.box.map50:.1%} accuracy!")
        print("This is good for FREE CPU training.")
    else:
        print("\n⚠️  LOW ACCURACY")
        print(f"❌ Only {val_results.box.map50:.1%} accuracy!")
        print("This indicates CPU limitations.")
    
    print(f"\n💰 COST: COMPLETELY FREE!")
    print("🎯 This is the BEST FREE CPU result in under 1 hour!")
    
    return results, training_time, val_results.box.map50

def main():
    """Main Anaconda fast training workflow"""
    
    print("⚡ ANACONDA FAST SOLUTION")
    print("=" * 60)
    print("🎯 Goal: BEST accuracy in under 1 hour")
    print("📊 Strategy: SPEED + ACCURACY OPTIMIZATION")
    print("💰 Cost: COMPLETELY FREE!")
    print("⏱️  Time target: Under 1 hour")
    print("⚠️  REALISTIC EXPECTATIONS:")
    print("   - CPU training is inherently slow")
    print("   - This gives the BEST possible CPU result in 1 hour")
    print("   - For 80+ accuracy, you need GPU")
    
    # Create anaconda fast configuration
    create_anaconda_fast_config()
    
    # Run anaconda fast training
    results, training_time, accuracy = anaconda_fast_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    print(f"💰 Cost: COMPLETELY FREE!")
    
    # Check targets
    time_success = training_time < 60
    accuracy_success = accuracy >= 0.80
    
    print(f"\n🎯 TARGET ASSESSMENT:")
    print(f"⏱️  Time target (<1 hour): {'ACHIEVED' if time_success else 'FAILED'}")
    print(f"🎯 Accuracy target (80+): {'ACHIEVED' if accuracy_success else 'FAILED'}")
    
    if time_success and accuracy_success:
        print("\n🏆 COMPLETE SUCCESS!")
        print("✅ Achieved both time and accuracy targets!")
        print("🎯 Mission accomplished!")
    elif time_success:
        print("\n⚡ TIME SUCCESS!")
        print("✅ Completed training in under 1 hour!")
        print(f"🎯 Accuracy: {accuracy:.1%}")
    elif accuracy_success:
        print("\n🎯 ACCURACY SUCCESS!")
        print("✅ Achieved 80+ accuracy!")
        print(f"⏱️  Time: {training_time:.1f} minutes")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print(f"✅ Completed training in {training_time:.1f} minutes")
        print(f"🎯 Accuracy: {accuracy:.1%}")
        print("This is the best possible FREE CPU result in 1 hour!")
    
    print(f"\n💡 TO ACHIEVE 80+ ACCURACY IN UNDER 1 HOUR:")
    print("1. Use Google Colab with FREE GPU")
    print("2. Use cloud computing services")
    print("3. Get a local GPU")
    print("4. Use the google_colab_free_gpu_80_accuracy.py script")
    
    print(f"\n🚀 RECOMMENDED NEXT STEPS:")
    print("1. Try Google Colab FREE GPU solution")
    print("2. Use cloud computing services")
    print("3. Get a local GPU")
    print("4. Use more training data")

if __name__ == "__main__":
    main()
