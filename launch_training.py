"""
ULTRA-FAST TRAINING LAUNCHER
============================
This script launches the ultra-fast training with real-time monitoring
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_ultra_fast_training():
    """Run the ultra-fast training with monitoring"""
    
    print("🚀 LAUNCHING ULTRA-FAST TRAINING")
    print("=" * 50)
    print("🎯 Target: 90+ accuracy in under 30 minutes")
    print("⚡ Starting training...")
    
    start_time = time.time()
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Run the ultra-fast training
        result = subprocess.run([
            sys.executable, "ultra_fast_training.py"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Total execution time: {training_time:.1f} minutes")
        print(f"📊 Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\n📝 Output:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print("\n📝 Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 30 minutes")
    except Exception as e:
        print(f"❌ Error running training: {e}")
    
    print(f"\n🎯 Mission {'ACCOMPLISHED' if training_time < 30 else 'FAILED'}")
    print(f"⏱️  Time taken: {training_time:.1f} minutes")

if __name__ == "__main__":
    run_ultra_fast_training()
