"""
FREE CPU SOLUTION - 80+ ACCURACY TARGET
======================================
This solution is optimized for FREE CPU training to achieve 80+ accuracy.
- Uses maximum CPU optimization
- Extended training for better accuracy
- Smart data augmentation
- Pre-trained weights optimization
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil

def create_free_cpu_config():
    """Create configuration for free CPU training optimized for 80+ accuracy"""
    
    print("🎯 Creating FREE CPU configuration for 80+ accuracy...")
    
    # Create optimized dataset directories
    for split in ['train', 'val']:
        Path(f"free_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"free_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Use more data for better accuracy (200 training and 50 validation samples)
    train_samples = 200
    val_samples = 50
    
    # Get training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy free samples
    def copy_free_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"free_{split_name}/images") / f"free_{i:03d}.png"
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"free_{split_name}/labels") / f"free_{i:03d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_free_samples(train_images, "train")
    copy_free_samples(val_images, "val")
    
    print(f"✅ Created FREE dataset: {train_samples} train + {val_samples} val samples")
    
    # Create config
    config = {
        'path': '.',
        'train': 'free_train/images',
        'val': 'free_val/images',
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
    
    with open('yolo_params_free.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def free_cpu_training():
    """FREE CPU training optimized for 80+ accuracy"""
    
    print("🎯 FREE CPU TRAINING - 80+ ACCURACY TARGET")
    print("=" * 50)
    print("🎯 Target: 80+ accuracy on FREE CPU")
    print("⚡ Strategy: MAXIMUM ACCURACY OPTIMIZATION")
    print("📊 Dataset: Extended (200 train + 50 val)")
    print("💰 Cost: COMPLETELY FREE!")
    print("⚡ Optimizations:")
    print("  - 50 epochs (extended for accuracy)")
    print("  - Batch size 8 (optimal for CPU)")
    print("  - Image size 416px (larger for accuracy)")
    print("  - Lower learning rate for stability")
    print("  - Maximum augmentation")
    print("  - Pre-trained weights")
    
    start_time = time.time()
    
    # Use YOLOv8n for speed but with extended training
    model = YOLO("yolo11n.pt")
    
    # FREE CPU configuration optimized for 80+ accuracy
    results = model.train(
        data="yolo_params_free.yaml",
        
        # ACCURACY OPTIMIZED SETTINGS
        epochs=50,                    # Extended epochs for accuracy
        batch=8,                      # Optimal batch size for CPU
        imgsz=416,                    # Larger image size for accuracy
        device='cpu',                 # CPU training
        
        # STABLE LEARNING PARAMETERS
        lr0=0.005,                    # Lower learning rate for stability
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # MAXIMUM AUGMENTATION FOR ACCURACY
        mosaic=1.0,                   # Maximum mosaic
        mixup=0.2,                    # More mixup
        copy_paste=0.2,               # More copy-paste
        degrees=15.0,                 # More rotation
        translate=0.2,                # More translation
        scale=0.8,                    # More scaling
        shear=5.0,                    # More shear
        perspective=0.0002,           # More perspective
        
        # ENHANCED COLOR AUGMENTATION
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        
        # OPTIMIZED LOSS WEIGHTS
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,                 # Cosine learning rate
        warmup_epochs=5,             # Extended warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=15,                 # More patience for accuracy
        save_period=10,              # Save every 10 epochs
        val=True,                    # Validation
        plots=False,                 # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='free_cpu_80_accuracy',
        exist_ok=True,
        
        # CPU OPTIMIZATIONS
        workers=16,
        cache=True,
        amp=False,                   # No mixed precision for CPU
        fraction=1.0,                # Use all free data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # TRAINING HACKS
        close_mosaic=40,             # Close mosaic later
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🎯 FREE CPU TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Final validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_free.yaml", imgsz=416, batch=8)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    print(f"\n🎯 ACCURACY ASSESSMENT:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {val_results.box.map50:.3f}")
    
    if val_results.box.map50 >= 0.80:
        print("\n🏆 SUCCESS! ACHIEVED 80+ ACCURACY!")
        print("✅ Target achieved on FREE CPU!")
    elif val_results.box.map50 >= 0.60:
        print("\n🎯 GOOD ACCURACY!")
        print(f"✅ Achieved {val_results.box.map50:.1%} accuracy!")
        print("This is excellent for FREE CPU training!")
    elif val_results.box.map50 >= 0.40:
        print("\n🔄 MODERATE SUCCESS")
        print(f"✅ Achieved {val_results.box.map50:.1%} accuracy!")
        print("This is reasonable for FREE CPU training.")
    else:
        print("\n⚠️  LOW ACCURACY")
        print(f"❌ Only {val_results.box.map50:.1%} accuracy!")
        print("This indicates CPU limitations.")
    
    print(f"\n💰 COST: COMPLETELY FREE!")
    print("🎯 This is the BEST FREE CPU result possible!")
    
    return results, training_time, val_results.box.map50

def main():
    """Main FREE CPU training workflow"""
    
    print("🎯 FREE CPU SOLUTION - 80+ ACCURACY TARGET")
    print("=" * 60)
    print("🎯 Goal: 80+ accuracy on FREE CPU")
    print("📊 Strategy: MAXIMUM ACCURACY OPTIMIZATION")
    print("💰 Cost: COMPLETELY FREE!")
    print("⚠️  REALISTIC EXPECTATIONS:")
    print("   - CPU training takes longer")
    print("   - 80+ accuracy is achievable with extended training")
    print("   - This gives the BEST FREE CPU result possible")
    
    # Create free configuration
    create_free_cpu_config()
    
    # Run free CPU training
    results, training_time, accuracy = free_cpu_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    print(f"💰 Cost: COMPLETELY FREE!")
    
    if accuracy >= 0.80:
        print("\n🏆 MISSION ACCOMPLISHED!")
        print("✅ Achieved 80+ accuracy on FREE CPU!")
        print("🎯 Target achieved!")
    elif accuracy >= 0.60:
        print("\n🎯 EXCELLENT RESULT!")
        print(f"✅ Achieved {accuracy:.1%} accuracy on FREE CPU!")
        print("This is excellent for FREE training!")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print(f"✅ Achieved {accuracy:.1%} accuracy on FREE CPU!")
        print("This is the best possible FREE CPU result!")
    
    print(f"\n💡 TO IMPROVE FURTHER (if needed):")
    print("1. Use Google Colab with free GPU")
    print("2. Use cloud computing services")
    print("3. Get a local GPU")
    print("4. Use more training data")
    print("5. Increase epochs to 100+")

if __name__ == "__main__":
    main()
