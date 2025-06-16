# 🎭 Face Shape Detection & Style Recommender - Quick Start Guide

## 🚀 Getting Started

### 1. Dependencies Installation
```bash
# Install required packages
pip install numpy opencv-python scikit-learn matplotlib Pillow streamlit
```

### 2. Training the Model
```bash
# Option 1: Using the training script
python lightweight_detector.py

# Option 2: Using the web interface
streamlit run simple_streamlit_app.py
# Then go to "Train Model" tab
```

### 3. Using the Application

#### 🌐 Web Interface (Recommended)
```bash
streamlit run simple_streamlit_app.py
# Open: http://localhost:8501
```

#### 🖥️ Command Line Interface
```bash
python simple_detector.py
# Choose: 1 for image file, 2 for camera
```

#### 📊 Batch Processing
```bash
python batch_processor.py "path/to/image/folder"
```

## 🎯 Usage Examples

### Single Image Detection
```python
from lightweight_detector import LightweightFaceShapeDetector

# Load model
detector = LightweightFaceShapeDetector()
detector.load_model('lightweight_face_shape_model.pkl')

# Predict
face_shape, confidence = detector.predict_face_shape('image.jpg')
print(f"Face Shape: {face_shape} (Confidence: {confidence:.2%})")
```

### Camera Detection
```python
from simple_detector import SimpleCameraDetector

# Initialize and start camera
detector = SimpleCameraDetector()
detector.detect_from_camera()  # Press 'q' to quit
```

## 📊 Project Structure

```
AI Project/
├── 📁 FaceShape Dataset/        # Training/testing images
├── 🧠 lightweight_detector.py   # Core ML model
├── 📹 simple_detector.py        # Camera interface
├── 🌐 simple_streamlit_app.py   # Web interface
├── 📊 batch_processor.py        # Batch processing
├── 📋 requirements.txt          # Dependencies
└── 📖 README.md                # Documentation
```

## 🎨 Face Shape Categories

| Shape | Description | Key Features |
|-------|-------------|--------------|
| 💖 Heart | Wider forehead, narrow chin | High cheekbones, pointed chin |
| 🥚 Oval | Well-balanced proportions | Length 1.5x width, soft curves |
| 🌙 Round | Equal width and height | Soft jawline, full cheeks |
| ⬛ Square | Strong angular jawline | Sharp features, wide forehead |
| 📏 Oblong | Longer than wide | High forehead, narrow features |

## 💇 Style Recommendations

### Heart Shape 💖
- **Hair**: Side-swept bangs, long layers, chin-length bob
- **Beard**: Light stubble, goatee, soul patch

### Oval Shape 🥚
- **Hair**: Most styles work, shoulder-length cuts, pixie cuts
- **Beard**: Full beard, stubble, mustache, clean shaven

### Round Shape 🌙
- **Hair**: Layered cuts, side parts, angular styles
- **Beard**: Angular beards, goatee, extended goatee

### Square Shape ⬛
- **Hair**: Soft layers, side-swept styles, wavy textures
- **Beard**: Rounded beards, circle beard, light stubble

### Oblong Shape 📏
- **Hair**: Bangs, layered bob, wide voluminous styles
- **Beard**: Full beard, mutton chops, wide mustache

## 🔧 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Solution: Run training first (`python lightweight_detector.py`)

2. **Low accuracy results**
   - Use clear, well-lit photos
   - Ensure face is unobstructed
   - Try different angles

3. **Camera not working**
   - Check camera permissions
   - Try different camera index
   - Ensure no other apps are using camera

4. **Web app not loading**
   - Check if port 8501 is available
   - Try: `streamlit run simple_streamlit_app.py --server.port 8502`

### Performance Tips

- **Better Images**: Use high-resolution, well-lit photos
- **Face Position**: Center the face in the image
- **Accessories**: Remove glasses, hats that obscure face shape
- **Expression**: Use neutral expressions for best results

## 📈 Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 4100+ dimensional feature vector
- **Accuracy**: 60-80% (varies by image quality)
- **Training Time**: 2-5 minutes
- **Prediction Time**: <1 second per image

## 🔄 Model Improvement Tips

1. **More Data**: Add more training images per category
2. **Better Features**: Implement facial landmark detection
3. **Deep Learning**: Use CNN for better accuracy
4. **Data Quality**: Ensure clean, labeled dataset

## 📞 Support

### File Structure Check
```bash
# Verify your project structure
dir /s "AI Project"  # Windows
ls -la "AI Project"  # Linux/Mac
```

### Model Status Check
```python
import os
print("Model exists:", os.path.exists('lightweight_face_shape_model.pkl'))
```

## 🎉 Features Implemented

- ✅ Face shape detection (5 categories)
- ✅ Style recommendations (hair & beard)
- ✅ Web interface with Streamlit
- ✅ Camera integration
- ✅ Batch processing
- ✅ Detailed reporting
- ✅ Export results (CSV, HTML)
- ✅ Visual statistics

## 🚧 Future Enhancements

- [ ] Deep learning model (CNN/ResNet)
- [ ] Facial landmark detection
- [ ] Real-time style preview
- [ ] Mobile app version
- [ ] API endpoint
- [ ] More style categories

---

**Made with ❤️ using Python, OpenCV, scikit-learn, and Streamlit**

For questions or issues, check the README.md file or review the code comments.
