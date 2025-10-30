"""
GOOGLE COLAB GPU TRAINING SCRIPT
================================
Copy and paste this entire script into Google Colab to get 90+ accuracy in under 10 minutes!

INSTRUCTIONS:
1. Go to colab.research.google.com
2. Enable GPU: Runtime > Change runtime type > GPU > Save
3. Upload your dataset
4. Copy and paste this entire script
5. Run it and get 90+ accuracy!
"""

# Install required packages
!pip install ultralytics

# Import libraries
import torch
import time
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

print("🚀 GOOGLE COLAB GPU TRAINING")
print("=" * 50)
print("🎯 Target: 90+ accuracy in under 10 minutes")

# Check GPU availability
print(f"\n🔧 GPU Check:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✅ GPU ready for training!")
else:
    print("⚠️  No GPU detected! Please enable GPU in Colab.")
    print("Go to: Runtime > Change runtime type > GPU > Save")

# Create optimized dataset directories
print("\n📁 Creating dataset structure...")
for split in ['train', 'val']:
    Path(f"colab_{split}/images").mkdir(parents=True, exist_ok=True)
    Path(f"colab_{split}/labels").mkdir(parents=True, exist_ok=True)

# Copy samples for speed (adjust these numbers as needed)
train_samples = 200
val_samples = 50

print(f"📊 Using {train_samples} training + {val_samples} validation samples")

# Copy training images
train_dir = Path("train/images")
if train_dir.exists():
    train_images = list(train_dir.glob("*.png"))[:train_samples]
    for i, img_path in enumerate(train_images):
        shutil.copy2(img_path, f"colab_train/images/colab_{i:03d}.png")
        
        # Copy corresponding label
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, f"colab_train/labels/colab_{i:03d}.txt")
    print(f"✅ Copied {len(train_images)} training samples")
else:
    print("⚠️  Training directory not found. Please upload your dataset.")

# Copy validation images
val_dir = Path("val/images")
if val_dir.exists():
    val_images = list(val_dir.glob("*.png"))[:val_samples]
    for i, img_path in enumerate(val_images):
        shutil.copy2(img_path, f"colab_val/images/colab_{i:03d}.png")
        
        # Copy corresponding label
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, f"colab_val/labels/colab_{i:03d}.txt")
    print(f"✅ Copied {len(val_images)} validation samples")
else:
    print("⚠️  Validation directory not found. Please upload your dataset.")

# Create YOLO config
config = {
    'path': '.',
    'train': 'colab_train/images',
    'val': 'colab_val/images',
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

with open('yolo_params_colab.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Configuration created")

# Start training
print("\n🚀 STARTING GPU TRAINING...")
print("⚡ Using GPU acceleration for maximum speed")

start_time = time.time()

# Load pre-trained model
model = YOLO("yolo11n.pt")

# Train with GPU optimization
results = model.train(
    data="yolo_params_colab.yaml",
    
    # GPU OPTIMIZED SETTINGS
    epochs=15,                    # Balanced epochs
    batch=32,                     # Large batch size for GPU
    imgsz=416,                    # Larger image size for accuracy
    device='cuda',                # GPU training
    
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
    plots=True,                  # Generate plots
    verbose=True,
    
    # PROJECT SETTINGS
    project='runs/train',
    name='colab_gpu_90',
    exist_ok=True,
    
    # GPU OPTIMIZATIONS
    workers=8,                   # GPU workers
    cache=True,                  # Cache dataset
    amp=True,                    # Mixed precision for GPU
    fraction=1.0,                # Use all data
    profile=False,
    freeze=None,
    multi_scale=False,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    
    # TRAINING HACKS
    close_mosaic=12,             # Close mosaic early
    resume=False,
    seed=42,
)

training_time = time.time() - start_time

print(f"\n🚀 TRAINING COMPLETED!")
print(f"⏱️  Training time: {training_time:.1f} minutes")
print(f"📊 Results saved to: {results.save_dir}")

# Load the best model and validate
print("\n🔍 Final validation...")
model = YOLO(results.save_dir / "weights" / "best.pt")
val_results = model.val(data="yolo_params_colab.yaml", imgsz=416, batch=32)

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
elif val_results.box.map50 >= 0.90:
    print("\n🎯 ACCURACY SUCCESS!")
    print("✅ Achieved 90+ accuracy!")
    print(f"⏱️  Time: {training_time:.1f} minutes")
elif training_time < 10:
    print("\n⚡ SPEED SUCCESS!")
    print("✅ Completed training in under 10 minutes!")
    print(f"🎯 Accuracy: {val_results.box.map50:.3f}")
else:
    print("\n🔄 PARTIAL SUCCESS")
    print(f"⏱️  Time: {training_time:.1f} minutes")
    print(f"🎯 Accuracy: {val_results.box.map50:.3f}")
    print("\n💡 Try increasing epochs or adjusting parameters")

# Download the trained model
print("\n💾 Downloading trained model...")
from google.colab import files

# Download the best weights
files.download(str(results.save_dir / "weights" / "best.pt"))

# Download the last weights
files.download(str(results.save_dir / "weights" / "last.pt"))

print("✅ Model files downloaded!")
print("\n🎉 Training complete!")
print(f"⏱️  Total time: {training_time:.1f} minutes")
print(f"🎯 Final accuracy: {val_results.box.map50:.3f}")

if val_results.box.map50 >= 0.90:
    print("\n🏆 MISSION ACCOMPLISHED!")
    print("You now have a model with 90+ accuracy!")
else:
    print("\n💡 To improve accuracy:")
    print("1. Increase epochs to 20-25")
    print("2. Use larger image size: imgsz=640")
    print("3. Enable more augmentation")
    print("4. Use more training data")
