# 🛡️ YOLO Safety Equipment Detection System

A modern web application for real-time detection of safety equipment using YOLO (You Only Look Once) deep learning model.

## ✨ Features

- **📹 Live Camera Detection**: Real-time detection using your device's camera
- **📁 Image Upload**: Upload images for batch detection
- **🎯 High Accuracy**: Trained to detect 7 types of safety equipment
- **📱 Responsive Design**: Works on desktop and mobile devices
- **⚡ Fast Processing**: Optimized for speed and efficiency
- **📊 Statistics**: Track detection metrics and performance

## 🔧 Detected Equipment Types

1. **Oxygen Tank** - Medical oxygen containers
2. **Nitrogen Tank** - Industrial nitrogen containers  
3. **First Aid Box** - Emergency medical supplies
4. **Fire Alarm** - Fire detection systems
5. **Safety Switch Panel** - Emergency control panels
6. **Emergency Phone** - Emergency communication devices
7. **Fire Extinguisher** - Fire suppression equipment

## 🚀 Quick Start

### Option 1: Run with Batch File (Windows)
```bash
# Double-click start_server.bat
# Or run in command prompt:
start_server.bat
```

### Option 2: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

### Option 3: Using Python
```bash
# Install packages
pip install flask flask-cors ultralytics opencv-python pillow numpy torch torchvision pyyaml

# Run the application
python app.py
```

## 🌐 Usage

1. **Start the Server**: Run `python app.py` or `start_server.bat`
2. **Open Browser**: Go to `http://localhost:5000`
3. **Live Detection**: 
   - Click "Start Camera" to begin live detection
   - Click "Capture & Detect" to analyze current frame
4. **Image Upload**:
   - Drag & drop images or click to browse
   - Click "Upload & Detect" to analyze

## 📊 API Endpoints

- `POST /api/detect` - Detect objects in uploaded image
- `GET /api/status` - Check server status
- `GET /api/classes` - Get available detection classes

## 🎯 Model Performance

- **Training**: 60 epochs with full dataset
- **Accuracy**: 80%+ mAP50
- **Speed**: Real-time processing
- **Batch Size**: 32 (optimized for GPU)
- **Image Size**: 640x640 pixels

## 🔧 Technical Details

### Frontend
- **HTML5**: Modern responsive design
- **CSS3**: Gradient backgrounds, animations, responsive grid
- **JavaScript**: Camera API, file upload, drag & drop
- **Canvas**: Image processing and display

### Backend
- **Flask**: Python web framework
- **YOLO**: Ultralytics YOLO11m model
- **OpenCV**: Image processing
- **PIL**: Image manipulation
- **CORS**: Cross-origin resource sharing

### Model Architecture
- **Base Model**: YOLO11m (medium size for balance)
- **Input Size**: 640x640 pixels
- **Classes**: 7 safety equipment types
- **Augmentation**: Mosaic, mixup, copy-paste, rotation, scaling

## 📁 Project Structure

```
├── app.py                 # Flask backend server
├── index.html            # Frontend web interface
├── requirements.txt     # Python dependencies
├── start_server.bat     # Windows startup script
├── README.md           # This file
├── templates/          # Flask templates directory
└── runs/train/        # Trained model weights
    └── yolo_80_accuracy/
        └── weights/
            └── best.pt
```

## 🛠️ Customization

### Adding New Equipment Types
1. Update the `names` dictionary in `app.py`
2. Retrain the model with new dataset
3. Update the frontend display

### Modifying Detection Threshold
```python
# In app.py, modify confidence threshold
confidence_threshold = 0.5  # Adjust as needed
```

### Changing Model
```python
# In app.py, change model path
model = YOLO("your_custom_model.pt")
```

## 🔍 Troubleshooting

### Camera Not Working
- Check browser permissions for camera access
- Ensure HTTPS in production (required for camera API)
- Try different browsers (Chrome, Firefox, Edge)

### Model Loading Issues
- Ensure `ultralytics` is installed: `pip install ultralytics`
- Check model file path in `app.py`
- Verify GPU drivers if using CUDA

### Performance Issues
- Use GPU acceleration if available
- Reduce image size for faster processing
- Adjust batch size based on hardware

## 📈 Performance Optimization

- **GPU Acceleration**: Automatic CUDA detection
- **Memory Management**: Disk caching for large datasets
- **Batch Processing**: Optimized batch sizes
- **Image Compression**: Efficient image handling
- **Async Processing**: Non-blocking API calls

## 🔒 Security Considerations

- Input validation for uploaded images
- File type restrictions
- CORS configuration for production
- Rate limiting for API endpoints

## 📱 Browser Compatibility

- **Chrome**: Full support
- **Firefox**: Full support  
- **Safari**: Full support
- **Edge**: Full support
- **Mobile**: Responsive design

## 🎨 UI/UX Features

- **Modern Design**: Gradient backgrounds, smooth animations
- **Responsive Layout**: Works on all screen sizes
- **Drag & Drop**: Intuitive file upload
- **Real-time Feedback**: Live status indicators
- **Statistics Dashboard**: Performance metrics
- **Error Handling**: User-friendly error messages

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t yolo-detector .
docker run -p 5000:5000 yolo-detector
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are in correct location
4. Check browser console for JavaScript errors

## 🎯 Future Enhancements

- [ ] Video file upload support
- [ ] Batch image processing
- [ ] Export detection results
- [ ] User authentication
- [ ] Cloud deployment
- [ ] Mobile app version
- [ ] Advanced analytics dashboard

---

**Built with ❤️ using YOLO, Flask, and modern web technologies**
