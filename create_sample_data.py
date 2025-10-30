"""
Create Sample Dataset for Space Station Safety Equipment Detection
This script creates sample images and labels for testing YOLO training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random

def create_sample_images():
    """Create sample images with bounding boxes"""
    
    # Create sample images
    for split in ['train', 'val', 'test']:
        images_dir = Path(f"{split}/images")
        labels_dir = Path(f"{split}/labels")
        
        # Create 10 sample images for each split
        for i in range(10):
            # Create a random image (640x640)
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Add some random shapes to simulate safety equipment
            for _ in range(random.randint(1, 3)):
                # Random rectangle (simulating safety equipment)
                x1 = random.randint(50, 500)
                y1 = random.randint(50, 500)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                
                # Create corresponding label file
                label_file = labels_dir / f"image_{i:03d}.txt"
                
                # Convert to YOLO format (normalized center, width, height)
                center_x = (x1 + x2) / 2 / 640
                center_y = (y1 + y2) / 2 / 640
                width = (x2 - x1) / 640
                height = (y2 - y1) / 640
                
                # Random class (0-6)
                class_id = random.randint(0, 6)
                
                # Write label file
                with open(label_file, 'w') as f:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            # Save image
            img_file = images_dir / f"image_{i:03d}.jpg"
            cv2.imwrite(str(img_file), img)
    
    print("✅ Sample dataset created!")
    print("📁 Structure:")
    print("  train/images/ - 10 sample training images")
    print("  train/labels/ - 10 corresponding label files")
    print("  val/images/   - 10 sample validation images")
    print("  val/labels/   - 10 corresponding label files")
    print("  test/images/  - 10 sample test images")
    print("  test/labels/  - 10 corresponding label files")

if __name__ == "__main__":
    create_sample_images()

