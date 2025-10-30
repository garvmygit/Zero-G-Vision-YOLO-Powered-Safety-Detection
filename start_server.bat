@echo off
echo Starting YOLO Safety Equipment Detection System...
echo ================================================

echo Installing required packages...
pip install -r requirements.txt

echo Starting Flask server...
python app.py

pause



