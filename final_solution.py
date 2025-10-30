"""
FINAL SOLUTION - 90+ ACCURACY IN UNDER 10 MINUTES
================================================
This is the final working solution for Anaconda:
- Balanced parameters for speed AND accuracy
- Smart data sampling
- Moderate augmentation
- Pre-trained weights optimization
"""

import os
import time
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil

def create_final_config():
    """Create configuration for final solution"""
    
    print("🎯 Creating final solution configuration...")
    
    # Create smart dataset directories
    for split in ['train', 'val']:
        Path(f"final_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"final_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Use 150 training and 40 validation samples for balance
    train_samples = 150
    val_samples = 40
    
    # Get training images
    train_dir = Path("../../hackathon2_train_3/train_3/train3/images")
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    
    # Get validation images
    val_dir = Path("../../hackathon2_train_3/train_3/val3/images")
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    
    # Copy final samples
    def copy_final_samples(images, split_name):
        for i, img_path in enumerate(images):
            # Copy image
            img_dest = Path(f"final_{split_name}/images") / f"final_{i:03d}.png"
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_dest = Path(f"final_{split_name}/labels") / f"final_{i:03d}.txt"
                shutil.copy2(label_path, label_dest)
    
    copy_final_samples(train_images, "train")
    copy_final_samples(val_images, "val")
    
    print(f"✅ Created final dataset: {train_samples} train + {val_samples} val samples")
    
    # Create config
    config = {
        'path': '.',
        'train': 'final_train/images',
        'val': 'final_val/images',
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
    
    with open('yolo_params_final.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def final_solution_training():
    """Final solution training for 90+ accuracy in under 10 minutes"""
    
    print("🏆 FINAL SOLUTION TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 10 minutes")
    print("⚡ Strategy: BALANCED SPEED + ACCURACY")
    print("📊 Dataset: Smart sampled (150 train + 40 val)")
    print("⚡ Optimizations:")
    print("  - 12 epochs (balanced)")
    print("  - Batch size 16")
    print("  - Image size 320px")
    print("  - Moderate learning rate")
    print("  - Smart augmentation")
    print("  - Pre-trained weights")
    
    start_time = time.time()
    
    # Use YOLOv8n for speed
    model = YOLO("yolo11n.pt")
    
    # FINAL SOLUTION configuration
    results = model.train(
        data="yolo_params_final.yaml",
        
        # BALANCED SPEED SETTINGS
        epochs=12,                    # Balanced epochs
        batch=16,                     # Moderate batch size
        imgsz=320,                    # Moderate image size
        device='cpu',                 # CPU training
        
        # BALANCED LEARNING PARAMETERS
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
        name='final_solution_90',
        exist_ok=True,
        
        # BALANCED OPTIMIZATIONS
        workers=16,
        cache=True,
        amp=False,                   # No mixed precision for CPU
        fraction=1.0,                # Use all final data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # TRAINING HACKS
        close_mosaic=10,             # Close mosaic early
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🏆 FINAL TRAINING COMPLETED!")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Quick validation
    print("\n🔍 Quick validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data="yolo_params_final.yaml", imgsz=320, batch=16)
    
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
    """Main final solution workflow"""
    
    print("🏆 FINAL SOLUTION")
    print("=" * 60)
    print("🎯 Goal: 90+ accuracy in under 10 minutes")
    print("📊 Strategy: BALANCED SPEED + ACCURACY")
    print("⚡ For Anaconda environment")
    
    # Create final configuration
    create_final_config()
    
    # Run final training
    results, training_time, accuracy = final_solution_training()
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"⏱️  Training time: {training_time:.1f} minutes")
    print(f"🎯 Final accuracy: {accuracy:.3f}")
    print(f"📁 Results: {results.save_dir}")
    
    if accuracy >= 0.90 and training_time < 10:
        print("\n🏆 COMPLETE SUCCESS!")
        print("✅ Achieved both time and accuracy targets!")
        print("\n🚀 To run in Anaconda:")
        print("1. Open Anaconda Prompt")
        print("2. Navigate to this directory")
        print("3. Run: python final_solution.py")
        print("4. If you have GPU, change device='cpu' to device='cuda'")
    elif accuracy >= 0.90:
        print("\n🎯 ACCURACY SUCCESS!")
        print("✅ Achieved 90+ accuracy!")
        print("\n💡 To make it faster:")
        print("1. Use GPU: device='cuda'")
        print("2. Reduce epochs to 8-10")
        print("3. Increase batch size to 32")
    elif training_time < 10:
        print("\n⚡ SPEED SUCCESS!")
        print("✅ Completed training in under 10 minutes!")
        print("\n💡 To improve accuracy:")
        print("1. Increase epochs to 15-20")
        print("2. Use larger image size: imgsz=416")
        print("3. Enable more augmentation")
    else:
        print("\n🔄 PARTIAL SUCCESS")
        print("⚠️  For better results:")
        print("1. Use GPU if available: device='cuda'")
        print("2. Try Google Colab with free GPU")
        print("3. Use cloud computing services")
        print("4. Consider transfer learning")

if __name__ == "__main__":
    main()
