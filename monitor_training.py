#!/usr/bin/env python3
"""
Training Progress Monitor
"""

import os
import time
from pathlib import Path

def monitor_training():
    """Monitor training progress"""
    print("Monitoring YOLO training progress...")
    
    # Check for training results
    results_dir = Path("runs/train/optimized_80_plus_50min_full_dataset")
    
    if results_dir.exists():
        print(f"Training results directory found: {results_dir}")
        
        # Check for weights
        weights_dir = results_dir / "weights"
        if weights_dir.exists():
            print(f"Weights directory found: {weights_dir}")
            
            # List weight files
            weight_files = list(weights_dir.glob("*.pt"))
            if weight_files:
                print("Weight files found:")
                for wf in weight_files:
                    size_mb = wf.stat().st_size / (1024 * 1024)
                    print(f"  - {wf.name}: {size_mb:.1f} MB")
            else:
                print("No weight files found yet")
        
        # Check for results
        results_file = results_dir / "results.csv"
        if results_file.exists():
            print(f"Results file found: {results_file}")
            with open(results_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print(f"Training progress: {len(lines)-1} epochs completed")
                    print("Latest results:")
                    print(lines[-1].strip())
        else:
            print("Results file not found yet")
    else:
        print("Training results directory not found yet")
    
    # Check for any running processes
    print("\nChecking for running training processes...")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'yolo' in ' '.join(proc.info['cmdline']).lower():
                    print(f"Found YOLO process: PID {proc.info['pid']}")
            except:
                pass
    except ImportError:
        print("psutil not available for process monitoring")

if __name__ == "__main__":
    monitor_training()

