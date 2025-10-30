"""
Enhanced Data Generation for Higher Accuracy
Creates more diverse and realistic training data
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import math

def create_enhanced_dataset():
    """Create enhanced dataset with more variety and realism"""
    
    print("🎨 Creating Enhanced Dataset for Higher Accuracy")
    print("=" * 50)
    
    # Create more samples for each split
    samples_per_split = 50  # Increased from 10
    
    for split in ['train', 'val', 'test']:
        images_dir = Path(f"{split}/images")
        labels_dir = Path(f"{split}/labels")
        
        print(f"📁 Creating {samples_per_split} samples for {split}...")
        
        for i in range(samples_per_split):
            # Create more realistic background
            img = create_realistic_background()
            
            # Add multiple objects per image
            num_objects = random.randint(1, 5)
            label_lines = []
            
            for obj_idx in range(num_objects):
                # Create realistic safety equipment
                obj_img, bbox, class_id = create_realistic_safety_equipment()
                
                # Place object on background
                img = place_object_on_background(img, obj_img, bbox)
                
                # Convert to YOLO format
                center_x, center_y, width, height = bbox
                center_x_norm = center_x / 640
                center_y_norm = center_y / 640
                width_norm = width / 640
                height_norm = height / 640
                
                label_lines.append(f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Save image
            img_file = images_dir / f"enhanced_{i:03d}.jpg"
            cv2.imwrite(str(img_file), img)
            
            # Save labels
            label_file = labels_dir / f"enhanced_{i:03d}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))
    
    print("✅ Enhanced dataset created!")
    print(f"📊 Total samples: {samples_per_split * 3}")
    print(f"📁 Structure:")
    print(f"  train/images/ - {samples_per_split} enhanced training images")
    print(f"  val/images/   - {samples_per_split} enhanced validation images")
    print(f"  test/images/  - {samples_per_split} enhanced test images")

def create_realistic_background():
    """Create realistic space station background"""
    
    # Create base background
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Add metallic floor
    cv2.rectangle(img, (0, 500), (640, 640), (80, 80, 80), -1)
    
    # Add wall panels
    for i in range(0, 640, 80):
        cv2.rectangle(img, (i, 0), (i+80, 500), (60, 60, 60), -1)
        cv2.rectangle(img, (i+2, 2), (i+78, 498), (70, 70, 70), -1)
    
    # Add lighting effects
    center_x, center_y = 320, 200
    for radius in range(200, 0, -20):
        intensity = max(0, 255 - radius)
        cv2.circle(img, (center_x, center_y), radius, (intensity, intensity, intensity), -1)
    
    # Add some noise for realism
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def create_realistic_safety_equipment():
    """Create realistic safety equipment objects"""
    
    # Choose random equipment type
    equipment_types = [
        ("oxygen_tank", (0, 0, 255)),      # Red oxygen tank
        ("nitrogen_tank", (0, 255, 0)),     # Green nitrogen tank
        ("first_aid", (255, 0, 0)),        # Blue first aid box
        ("fire_alarm", (0, 255, 255)),      # Yellow fire alarm
        ("safety_panel", (255, 0, 255)),    # Magenta safety panel
        ("emergency_phone", (255, 255, 0)), # Cyan emergency phone
        ("fire_extinguisher", (128, 0, 128)) # Purple fire extinguisher
    ]
    
    equipment_name, color = random.choice(equipment_types)
    class_id = equipment_types.index((equipment_name, color))
    
    # Create object based on type
    if equipment_name == "oxygen_tank":
        obj_img = create_oxygen_tank(color)
    elif equipment_name == "nitrogen_tank":
        obj_img = create_nitrogen_tank(color)
    elif equipment_name == "first_aid":
        obj_img = create_first_aid_box(color)
    elif equipment_name == "fire_alarm":
        obj_img = create_fire_alarm(color)
    elif equipment_name == "safety_panel":
        obj_img = create_safety_panel(color)
    elif equipment_name == "emergency_phone":
        obj_img = create_emergency_phone(color)
    else:  # fire_extinguisher
        obj_img = create_fire_extinguisher(color)
    
    # Random size
    scale = random.uniform(0.8, 1.5)
    new_width = int(obj_img.shape[1] * scale)
    new_height = int(obj_img.shape[0] * scale)
    obj_img = cv2.resize(obj_img, (new_width, new_height))
    
    # Random position
    x = random.randint(50, 640 - new_width - 50)
    y = random.randint(50, 640 - new_height - 50)
    
    bbox = (x + new_width//2, y + new_height//2, new_width, new_height)
    
    return obj_img, bbox, class_id

def create_oxygen_tank(color):
    """Create oxygen tank object"""
    img = np.zeros((80, 40, 3), dtype=np.uint8)
    
    # Tank body
    cv2.ellipse(img, (20, 40), (15, 30), 0, 0, 180, color, -1)
    cv2.ellipse(img, (20, 40), (15, 30), 0, 180, 360, (color[0]//2, color[1]//2, color[2]//2), -1)
    
    # Tank valve
    cv2.rectangle(img, (18, 10), (22, 20), (100, 100, 100), -1)
    
    return img

def create_nitrogen_tank(color):
    """Create nitrogen tank object"""
    img = np.zeros((80, 40, 3), dtype=np.uint8)
    
    # Tank body
    cv2.ellipse(img, (20, 40), (15, 30), 0, 0, 180, color, -1)
    cv2.ellipse(img, (20, 40), (15, 30), 0, 180, 360, (color[0]//2, color[1]//2, color[2]//2), -1)
    
    # Different valve for nitrogen
    cv2.rectangle(img, (17, 10), (23, 20), (150, 150, 150), -1)
    
    return img

def create_first_aid_box(color):
    """Create first aid box object"""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    
    # Box body
    cv2.rectangle(img, (5, 5), (75, 55), color, -1)
    cv2.rectangle(img, (5, 5), (75, 55), (0, 0, 0), 2)
    
    # Red cross
    cv2.rectangle(img, (35, 20), (45, 40), (0, 0, 255), -1)
    cv2.rectangle(img, (25, 30), (55, 40), (0, 0, 255), -1)
    
    return img

def create_fire_alarm(color):
    """Create fire alarm object"""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    
    # Alarm body
    cv2.circle(img, (30, 30), 25, color, -1)
    cv2.circle(img, (30, 30), 25, (0, 0, 0), 2)
    
    # Alarm light
    cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
    
    return img

def create_safety_panel(color):
    """Create safety panel object"""
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    
    # Panel body
    cv2.rectangle(img, (5, 5), (115, 75), color, -1)
    cv2.rectangle(img, (5, 5), (115, 75), (0, 0, 0), 2)
    
    # Buttons
    for i in range(3):
        for j in range(4):
            x = 15 + j * 25
            y = 15 + i * 20
            cv2.circle(img, (x, y), 8, (50, 50, 50), -1)
    
    return img

def create_emergency_phone(color):
    """Create emergency phone object"""
    img = np.zeros((60, 40, 3), dtype=np.uint8)
    
    # Phone body
    cv2.rectangle(img, (5, 5), (35, 55), color, -1)
    cv2.rectangle(img, (5, 5), (35, 55), (0, 0, 0), 2)
    
    # Screen
    cv2.rectangle(img, (8, 8), (32, 25), (0, 0, 0), -1)
    
    # Buttons
    cv2.circle(img, (20, 40), 5, (100, 100, 100), -1)
    
    return img

def create_fire_extinguisher(color):
    """Create fire extinguisher object"""
    img = np.zeros((100, 40, 3), dtype=np.uint8)
    
    # Extinguisher body
    cv2.rectangle(img, (15, 20), (25, 80), color, -1)
    cv2.rectangle(img, (15, 20), (25, 80), (0, 0, 0), 2)
    
    # Nozzle
    cv2.rectangle(img, (18, 10), (22, 20), (100, 100, 100), -1)
    
    # Handle
    cv2.rectangle(img, (5, 30), (15, 35), (150, 150, 150), -1)
    
    return img

def place_object_on_background(background, obj_img, bbox):
    """Place object on background with proper blending"""
    
    x, y, w, h = bbox
    x = x - w//2
    y = y - h//2
    
    # Ensure object fits within background
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > background.shape[1]: w = background.shape[1] - x
    if y + h > background.shape[0]: h = background.shape[0] - y
    
    # Resize object if needed
    if obj_img.shape[1] != w or obj_img.shape[0] != h:
        obj_img = cv2.resize(obj_img, (w, h))
    
    # Blend object with background
    alpha = 0.8  # Transparency
    background[y:y+h, x:x+w] = cv2.addWeighted(
        background[y:y+h, x:x+w], 1-alpha,
        obj_img, alpha, 0
    )
    
    return background

if __name__ == "__main__":
    create_enhanced_dataset()

