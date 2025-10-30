"""
CPU Training Script - Avoiding GPU Memory Issues
"""
import os
import subprocess
import sys
import time

def run_cpu_training():
    """Run training on CPU to avoid GPU memory issues"""
    
    print("Starting CPU training...")
    print("Target: 80%+ accuracy in 50 minutes")
    print("Parameters: 60 epochs, batch 8, image size 640px, CPU only")
    
    # Create training script for CPU
    training_script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

try:
    from ultralytics import YOLO
    import time
    
    print("Loading YOLO model...")
    model = YOLO('yolo11s.pt')
    
    print("Starting CPU training...")
    start_time = time.time()
    
    results = model.train(
        data='yolo_params_optimized.yaml',
        epochs=60,
        batch=8,  # Smaller batch for CPU
        imgsz=640,
        device='cpu',  # Force CPU
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
        name='cpu_optimized_80_plus',
        exist_ok=True,
        workers=4,  # Fewer workers for CPU
        cache=False,  # No caching for CPU
        amp=False,  # No mixed precision for CPU
        close_mosaic=50,
        seed=42
    )
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.1f} minutes")
    print(f"Results saved to: {results.save_dir}")
    
    # Quick validation
    model = YOLO(results.save_dir / "weights" / "best.pt")
    val_results = model.val(data='yolo_params_optimized.yaml', imgsz=640, batch=8, device='cpu')
    
    print(f"Final mAP50: {val_results.box.map50:.3f}")
    
    if val_results.box.map50 >= 0.80:
        print("SUCCESS: Achieved 80%+ accuracy!")
    else:
        print(f"Current accuracy: {val_results.box.map50:.1%}")
        
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open('cpu_training.py', 'w') as f:
        f.write(training_script)
    
    # Run the CPU training script
    try:
        print("Starting CPU training (this will take longer but should work)...")
        result = subprocess.run([sys.executable, 'cpu_training.py'], 
                              capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        print("Training output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("CPU Training completed successfully!")
        else:
            print("CPU Training failed!")
            
    except subprocess.TimeoutExpired:
        print("CPU Training timed out after 2 hours")
    except Exception as e:
        print(f"Error running CPU training: {e}")
    
    # Clean up
    if os.path.exists('cpu_training.py'):
        os.remove('cpu_training.py')

if __name__ == "__main__":
    run_cpu_training()

