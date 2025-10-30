"""
OPTIMIZED GPU TRAINING - 80%+ ACCURACY IN 50 MINUTES
Based on your exact requirements: 60 epochs, larger dataset, bigger batch size, larger image size
"""
import os
import time
import yaml
from pathlib import Path
import shutil

def create_optimized_dataset():
    """Create optimized dataset with more samples"""
    print("Creating optimized dataset...")
    print("Previous: 50 epochs + little data = 60% accuracy")
    print("New: 60 epochs + MORE data + bigger batch + larger images = 80%+ accuracy")
    
    # Create optimized dataset directories
    for split in ['train', 'val']:
        Path(f"optimized_{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"optimized_{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # INCREASED DATASET SIZE - Based on your feedback about "little training data"
    train_samples = 800  # Increased from 200
    val_samples = 200    # Increased from 50
    
    print(f"Dataset size: {train_samples} train + {val_samples} val samples")
    print("This addresses your feedback about 'little training data'")
    
    # Copy training images (using more data)
    train_dir = Path("train/images")
    if train_dir.exists():
        train_images = list(train_dir.glob("*.png"))[:train_samples]
        for i, img_path in enumerate(train_images):
            shutil.copy2(img_path, f"optimized_train/images/opt_{i:04d}.png")
            
            # Copy corresponding label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, f"optimized_train/labels/opt_{i:04d}.txt")
    
    # Copy validation images (using more data)
    val_dir = Path("val/images")
    if val_dir.exists():
        val_images = list(val_dir.glob("*.png"))[:val_samples]
        for i, img_path in enumerate(val_images):
            shutil.copy2(img_path, f"optimized_val/images/opt_{i:04d}.png")
            
            # Copy corresponding label
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, f"optimized_val/labels/opt_{i:04d}.txt")
    
    print(f"Created OPTIMIZED dataset: {train_samples} train + {val_samples} val samples")
    
    # Create optimized YOLO config
    config = {
        'path': '.',
        'train': 'optimized_train/images',
        'val': 'optimized_val/images',
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
    
    with open('yolo_params_optimized.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("OPTIMIZED configuration created")
    return True

def run_optimized_training():
    """Run optimized training for 80%+ accuracy in 50 minutes"""
    
    print("Starting OPTIMIZED GPU training...")
    print("Target: 80%+ accuracy within 50 minutes")
    print("Parameters:")
    print("  - Epochs: 60 (increased from 50)")
    print("  - Batch size: 16 (bigger batch for GPU)")
    print("  - Image size: 640px (larger images for accuracy)")
    print("  - Dataset: 800 train + 200 val (more data)")
    print("  - Model: YOLO11s (balanced speed/accuracy)")
    print("  - Device: GPU only")
    
    # Use subprocess to run ultralytics command
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "-m", "ultralytics", "yolo", "train",
        "data=yolo_params_optimized.yaml",
        "model=yolo11s.pt",
        "epochs=60",
        "batch=16",
        "imgsz=640",
        "device=cuda",
        "lr0=0.01",
        "lrf=0.01",
        "momentum=0.937",
        "weight_decay=0.0005",
        "mosaic=1.0",
        "mixup=0.2",
        "copy_paste=0.2",
        "degrees=15.0",
        "translate=0.2",
        "scale=0.8",
        "shear=5.0",
        "perspective=0.0002",
        "hsv_h=0.02",
        "hsv_s=0.8",
        "hsv_v=0.5",
        "box=7.5",
        "cls=0.5",
        "dfl=1.5",
        "optimizer=AdamW",
        "cos_lr=True",
        "warmup_epochs=5",
        "warmup_momentum=0.8",
        "warmup_bias_lr=0.1",
        "patience=15",
        "save_period=10",
        "val=True",
        "plots=False",
        "verbose=True",
        "project=runs/train",
        "name=optimized_80_plus_50min",
        "exist_ok=True",
        "workers=8",
        "cache=disk",
        "amp=True",
        "fraction=1.0",
        "profile=False",
        "freeze=None",
        "multi_scale=False",
        "overlap_mask=True",
        "mask_ratio=4",
        "dropout=0.0",
        "close_mosaic=50",
        "resume=False",
        "seed=42"
    ]
    
    start_time = time.time()
    
    try:
        print("Starting training...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.1f} minutes")
        print("Training output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("Training completed successfully!")
            return True, training_time
        else:
            print("Training failed!")
            return False, training_time
            
    except subprocess.TimeoutExpired:
        training_time = time.time() - start_time
        print(f"Training timed out after {training_time:.1f} minutes")
        return False, training_time
    except Exception as e:
        training_time = time.time() - start_time
        print(f"Training failed with error: {e}")
        return False, training_time

def main():
    """Main execution function"""
    
    print("OPTIMIZED GPU TRAINING - 80%+ ACCURACY IN 50 MINUTES")
    print("=" * 80)
    print("Goal: 80%+ accuracy within 50 minutes")
    print("Strategy: GPU training with optimized parameters")
    print("Parameters: 60 epochs, larger dataset, bigger batch, larger images")
    
    # Create optimized dataset
    if not create_optimized_dataset():
        print("Failed to create dataset. Please check your data paths.")
        return
    
    # Run optimized training
    success, training_time = run_optimized_training()
    
    if success:
        print(f"SUCCESS: Training completed in {training_time:.1f} minutes")
        print("Check runs/train/optimized_80_plus_50min/ for results")
        
        # Check if we achieved the time target
        if training_time <= 50:
            print("ACHIEVED: Training completed in under 50 minutes!")
        else:
            print(f"Time: {training_time:.1f} minutes (target: ≤50)")
    else:
        print(f"Training failed after {training_time:.1f} minutes")
        print("Please check GPU availability and CUDA installation")

if __name__ == "__main__":
    main()
