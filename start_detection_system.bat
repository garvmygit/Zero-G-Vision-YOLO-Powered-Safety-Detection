@echo off
echo Starting YOLO Safety Equipment Detection System...
echo ================================================

cd /d "C:\Users\garvg\OneDrive\Desktop\AI\Hackathon2_scripts\Hackathon2_scripts"

echo Installing required packages...
pip install flask flask-cors ultralytics opencv-python pillow numpy pyyaml

echo Starting Flask server...
echo.
echo The system will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python run_live_detection.py

pause

