
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

try:
    from ultralytics import YOLO
    
    print("Testing YOLO11s model...")
    model = YOLO('yolo11s.pt')
    
    # Test on validation data
    print("Running validation...")
    results = model.val(data='yolo_params_optimized.yaml', imgsz=640, batch=8, device='cpu')
    
    print(f"YOLO11s mAP50: {results.box.map50:.3f}")
    print(f"YOLO11s mAP50-95: {results.box.map:.3f}")
    print(f"YOLO11s Precision: {results.box.mp:.3f}")
    print(f"YOLO11s Recall: {results.box.mr:.3f}")
    
    if results.box.map50 >= 0.80:
        print("SUCCESS: YOLO11s achieved 80%+ accuracy!")
    else:
        print(f"YOLO11s accuracy: {results.box.map50:.1%}")
    
    print("\nTesting YOLO11m model...")
    model_m = YOLO('yolo11m.pt')
    results_m = model_m.val(data='yolo_params_optimized.yaml', imgsz=640, batch=4, device='cpu')
    
    print(f"YOLO11m mAP50: {results_m.box.map50:.3f}")
    print(f"YOLO11m mAP50-95: {results_m.box.map:.3f}")
    print(f"YOLO11m Precision: {results_m.box.mp:.3f}")
    print(f"YOLO11m Recall: {results_m.box.mr:.3f}")
    
    if results_m.box.map50 >= 0.80:
        print("SUCCESS: YOLO11m achieved 80%+ accuracy!")
    else:
        print(f"YOLO11m accuracy: {results_m.box.map50:.1%}")
        
except Exception as e:
    print(f"Testing failed: {e}")
    import traceback
    traceback.print_exc()

