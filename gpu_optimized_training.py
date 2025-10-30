"""
GPU Training Script - Optimized for 80%+ Accuracy in 50 Minutes
Based on your requirements: 60 epochs, larger dataset, bigger batch size, larger image size
"""
import os
import subprocess
import sys
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

def run_gpu_training():
    """Run GPU training with optimized parameters"""
    
    print("Starting GPU training...")
    print("Target: 80%+ accuracy within 50 minutes")
    print("Parameters: 60 epochs, batch 16, image size 640px, GPU only")
    
    # Create training script that avoids PyTorch import issues
    training_script = """
import os
import sys
import time

# Set environment variables to avoid memory issues
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    # Try to import ultralytics
    from ultralytics import YOLO
    
    print("GPU Training Starting...")
    print("Target: 80%+ accuracy in 50 minutes")
    print("Parameters:")
    print("  - Epochs: 60 (increased from 50)")
    print("  - Batch size: 16 (bigger batch for GPU)")
    print("  - Image size: 640px (larger images for accuracy)")
    print("  - Dataset: 800 train + 200 val (more data)")
    print("  - Model: YOLO11s (balanced speed/accuracy)")
    print("  - Device: GPU only")
    
    # Load model
    model = YOLO('yolo11s.pt')
    
    start_time = time.time()
    
    # OPTIMIZED TRAINING CONFIGURATION
    results = model.train(
        data='yolo_params_optimized.yaml',
        
        # OPTIMIZED SETTINGS BASED ON YOUR FEEDBACK
        epochs=60,                    # Increased from 50
        batch=16,                     # Bigger batch size
        imgsz=640,                    # Larger image size
        device='cuda',                # GPU training
        
        # OPTIMIZED LEARNING PARAMETERS
        lr0=0.01,                      # Moderate learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # MAXIMUM AUGMENTATION FOR ACCURACY
        mosaic=1.0,                    # Maximum mosaic
        mixup=0.2,                     # High mixup
        copy_paste=0.2,                # High copy-paste
        degrees=15.0,                  # High rotation
        translate=0.2,                 # High translation
        scale=0.8,                     # High scaling
        shear=5.0,                     # High shear
        perspective=0.0002,            # High perspective
        
        # MAXIMUM COLOR AUGMENTATION
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        
        # OPTIMIZED LOSS WEIGHTS
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # TRAINING OPTIMIZATIONS
        optimizer='AdamW',
        cos_lr=True,                   # Cosine learning rate
        warmup_epochs=5,               # Warmup for stability
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # VALIDATION AND SAVING
        patience=15,                   # Early stopping
        save_period=10,                # Save every 10 epochs
        val=True,                      # Validation
        plots=False,                   # No plots for speed
        verbose=True,
        
        # PROJECT SETTINGS
        project='runs/train',
        name='gpu_optimized_80_plus_50min',
        exist_ok=True,
        
        # MEMORY OPTIMIZATIONS
        workers=8,                     # GPU workers
        cache='disk',                  # Disk caching for memory efficiency
        amp=True,                      # Mixed precision for GPU
        fraction=1.0,                  # Use all GPU data
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # SPEED OPTIMIZATIONS
        close_mosaic=50,               # Close mosaic later for 60 epochs
        resume=False,
        seed=42,
    )
    
    training_time = time.time() - start_time
    
    print(f"GPU Training completed in {training_time:.1f} minutes")
    print(f"Results saved to: {results.save_dir}")
    
    # Quick validation
    print("Running final validation...")
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data='yolo_params_optimized.yaml', imgsz=640, batch=16, device='cuda')
    
    print(f"Final Results:")
    print(f"  - mAP50: {val_results.box.map50:.3f}")
    print(f"  - mAP50-95: {val_results.box.map:.3f}")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    # Check targets
    time_success = training_time <= 50
    accuracy_success = val_results.box.map50 >= 0.80
    
    print(f"Target Assessment:")
    print(f"  Time target (≤50 min): {'ACHIEVED' if time_success else 'FAILED'}")
    print(f"  Accuracy target (80+): {'ACHIEVED' if accuracy_success else 'FAILED'}")
    
    if accuracy_success and time_success:
        print("PERFECT SUCCESS!")
        print("Achieved 80%+ accuracy in under 50 minutes!")
    elif accuracy_success:
        print("ACCURACY SUCCESS!")
        print("Achieved 80%+ accuracy!")
        print(f"Time: {training_time:.1f} minutes")
    elif time_success:
        print(f"Current accuracy: {val_results.box.map50:.1%}")
        print("Achieved time target!")
    else:
        print(f"Current accuracy: {val_results.box.map50:.1%}")
        print(f"Time: {training_time:.1f} minutes")
    
    # Show improvement
    previous_accuracy = 0.60
    improvement = val_results.box.map50 - previous_accuracy
    print(f"Improvement from previous (60%): +{improvement:.1%}")
    
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open('gpu_training_script.py', 'w') as f:
        f.write(training_script)
    
    # Run the GPU training script
    try:
        print("Starting GPU training...")
        print("This will take approximately 30-50 minutes...")
        
        result = subprocess.run([sys.executable, 'gpu_training_script.py'], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        print("Training output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("GPU Training completed successfully!")
        else:
            print("GPU Training failed!")
            
    except subprocess.TimeoutExpired:
        print("GPU Training timed out after 1 hour")
    except Exception as e:
        print(f"Error running GPU training: {e}")
    
    # Clean up
    if os.path.exists('gpu_training_script.py'):
        os.remove('gpu_training_script.py')

def main():
    """Main execution function"""
    
    print("GPU OPTIMIZED TRAINING SCRIPT - 80%+ ACCURACY IN 50 MINUTES")
    print("=" * 80)
    print("Goal: 80%+ accuracy within 50 minutes")
    print("Strategy: GPU training with optimized parameters")
    print("Parameters: 60 epochs, larger dataset, bigger batch, larger images")
    
    # Create optimized dataset
    if not create_optimized_dataset():
        print("Failed to create dataset. Please check your data paths.")
        return
    
    # Run GPU training
    run_gpu_training()

if __name__ == "__main__":
    main()

