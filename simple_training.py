"""
Simple Training Script - Avoiding PyTorch Memory Issues
"""
import os
import subprocess
import sys

def run_training():
    """Run training using subprocess to avoid memory issues"""
    
    print("Starting optimized training...")
    print("Target: 80%+ accuracy in 50 minutes")
    print("Parameters: 60 epochs, batch 16, image size 640px")
    
    # First, let's create the dataset
    print("Creating optimized dataset...")
    
    # Create directories
    os.makedirs("optimized_train/images", exist_ok=True)
    os.makedirs("optimized_train/labels", exist_ok=True)
    os.makedirs("optimized_val/images", exist_ok=True)
    os.makedirs("optimized_val/labels", exist_ok=True)
    
    # Copy training data (simplified)
    import shutil
    from pathlib import Path
    
    train_dir = Path("train/images")
    if train_dir.exists():
        train_images = list(train_dir.glob("*.png"))[:800]  # 800 samples
        for i, img_path in enumerate(train_images):
            shutil.copy2(img_path, f"optimized_train/images/opt_{i:04d}.png")
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, f"optimized_train/labels/opt_{i:04d}.txt")
    
    val_dir = Path("val/images")
    if val_dir.exists():
        val_images = list(val_dir.glob("*.png"))[:200]  # 200 samples
        for i, img_path in enumerate(val_images):
            shutil.copy2(img_path, f"optimized_val/images/opt_{i:04d}.png")
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, f"optimized_val/labels/opt_{i:04d}.txt")
    
    print("Dataset created successfully!")
    
    # Create config file
    config = """path: .
train: optimized_train/images
val: optimized_val/images
nc: 7
names:
  0: OxygenTank
  1: NitrogenTank
  2: FirstAidBox
  3: FireAlarm
  4: SafetySwitchPanel
  5: EmergencyPhone
  6: FireExtinguisher"""
    
    with open('yolo_params_optimized.yaml', 'w') as f:
        f.write(config)
    
    print("Configuration created!")
    
    # Try to run training using a simple approach
    print("Starting training...")
    
    # Use a simple Python script to avoid memory issues
    training_script = """
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

try:
    from ultralytics import YOLO
    import time
    
    print("Loading YOLO model...")
    model = YOLO('yolo11s.pt')
    
    print("Starting training...")
    start_time = time.time()
    
    results = model.train(
        data='yolo_params_optimized.yaml',
        epochs=60,
        batch=16,
        imgsz=640,
        device='cuda',
        lr0=0.01,
        mosaic=1.0,
        mixup=0.2,
        degrees=15.0,
        translate=0.2,
        scale=0.8,
        shear=5.0,
        perspective=0.0002,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        optimizer='AdamW',
        cos_lr=True,
        warmup_epochs=5,
        patience=15,
        save_period=10,
        val=True,
        plots=False,
        verbose=True,
        project='runs/train',
        name='optimized_80_plus_50min',
        exist_ok=True,
        workers=8,
        cache='disk',
        amp=True,
        close_mosaic=50,
        seed=42
    )
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.1f} minutes")
    print(f"Results saved to: {results.save_dir}")
    
    # Quick validation
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data='yolo_params_optimized.yaml', imgsz=640, batch=16)
    
    print(f"Final mAP50: {val_results.box.map50:.3f}")
    
    if val_results.box.map50 >= 0.80:
        print("SUCCESS: Achieved 80%+ accuracy!")
    else:
        print(f"Current accuracy: {val_results.box.map50:.1%}")
        
except Exception as e:
    print(f"Training failed: {e}")
    print("This might be due to memory issues or GPU problems")
"""
    
    with open('temp_training.py', 'w') as f:
        f.write(training_script)
    
    # Run the training script
    try:
        result = subprocess.run([sys.executable, 'temp_training.py'], 
                              capture_output=True, text=True, timeout=3600)
        
        print("Training output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("Training completed successfully!")
        else:
            print("Training failed!")
            
    except subprocess.TimeoutExpired:
        print("Training timed out after 1 hour")
    except Exception as e:
        print(f"Error running training: {e}")
    
    # Clean up
    if os.path.exists('temp_training.py'):
        os.remove('temp_training.py')

if __name__ == "__main__":
    run_training()

