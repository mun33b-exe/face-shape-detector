# 🎭 Face Shape Detection & Style Recommender

An AI-powered application that detects face shapes from images or camera feed and provides personalized hairstyle and beard style recommendations.

## 🌟 Features

- **Accurate Face Shape Detection**: CNN model trained on your custom dataset
- **5 Face Shape Categories**: Heart, Oval, Round, Square, Oblong
- **Style Recommendations**: Personalized hairstyle and beard suggestions
- **Multiple Interfaces**: Web app, command line, and camera integration
- **Real-time Detection**: Live camera feed processing
- **High Accuracy**: Optimized CNN architecture with data augmentation

## 📁 Project Structure

```
AI Project/
├── FaceShape Dataset/          # Your training dataset
│   ├── training_set/
│   │   ├── Heart/
│   │   ├── Oval/
│   │   ├── Round/
│   │   ├── Square/
│   │   └── Oblong/
│   └── testing_set/
│       ├── Heart/
│       ├── Oval/
│       ├── Round/
│       ├── Square/
│       └── Oblong/
├── face_shape_detector.py     # Core ML model
├── camera_detector.py         # Camera integration
├── streamlit_app.py          # Web interface
├── train_model.py            # Training script
├── test_model.py             # Testing utilities
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess your dataset
- Create a CNN model architecture
- Train for 30 epochs with data augmentation
- Evaluate on test set
- Save the trained model as `face_shape_model.h5`

### 3. Test the Model

```bash
python test_model.py
```

Choose from:
- Test with image file
- Test with camera (real-time)

### 4. Run Web Interface

```bash
streamlit run streamlit_app.py
```

Access the web app at `http://localhost:8501`

## 🎯 Usage Examples

### Command Line Detection

```python
from face_shape_detector import FaceShapeDetector

# Load trained model
detector = FaceShapeDetector()
detector.load_model('face_shape_model.h5')

# Predict face shape
face_shape, confidence = detector.predict_face_shape('path/to/image.jpg')
print(f"Face Shape: {face_shape} (Confidence: {confidence:.2%})")
```

### Camera Integration

```python
from camera_detector import CameraFaceShapeDetector

# Initialize camera detector
detector = CameraFaceShapeDetector()

# Start real-time detection
detector.detect_from_camera()
```

## 🏗️ Model Architecture

- **Input Layer**: 224×224×3 RGB images
- **Convolutional Blocks**: 4 blocks with increasing filters (32→64→128→256)
- **Batch Normalization**: After each conv layer
- **Dropout**: 0.25 in conv blocks, 0.5 in dense layers
- **Global Average Pooling**: Reduces overfitting
- **Dense Layers**: 512→256→5 neurons
- **Output**: 5 classes with softmax activation

## 📊 Dataset Requirements

Your dataset should follow this structure:

```
FaceShape Dataset/
├── training_set/
│   ├── Heart/       # Heart-shaped faces
│   ├── Oblong/      # Oblong/rectangular faces
│   ├── Oval/        # Oval faces
│   ├── Round/       # Round faces
│   └── Square/      # Square faces
└── testing_set/
    ├── Heart/
    ├── Oblong/
    ├── Oval/
    ├── Round/
    └── Square/
```

**Recommended**: 500+ images per class for best results.

## 🎨 Style Recommendations

### Heart Shape
- **Hair**: Side-swept bangs, long layers, chin-length bob
- **Beard**: Light stubble, goatee, soul patch

### Oval Shape
- **Hair**: Most styles work, shoulder-length, pixie cut
- **Beard**: Full beard, stubble, mustache, clean shaven

### Round Shape
- **Hair**: Layered cuts, side part, angular styles
- **Beard**: Angular beard, goatee, extended goatee

### Square Shape
- **Hair**: Soft layers, side-swept, wavy styles
- **Beard**: Rounded beard, circle beard, light stubble

### Oblong Shape
- **Hair**: Bangs, layered bob, wide styles
- **Beard**: Full beard, mutton chops, wide mustache

## ⚙️ Configuration

### Training Parameters

```python
# In face_shape_detector.py
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (224, 224)
```

### Data Augmentation

- Rotation: ±10 degrees
- Width/Height shift: ±10%
- Horizontal flip: Yes
- Zoom: ±10%

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Run `python train_model.py` first

2. **Low accuracy**
   - Ensure balanced dataset (equal images per class)
   - Increase training epochs
   - Add more training data

3. **Camera not working**
   - Check camera permissions
   - Try different camera index (0, 1, 2)
   - Install proper OpenCV version

4. **Memory errors**
   - Reduce batch size
   - Use smaller image size
   - Close other applications

### Performance Tips

- **GPU Training**: Install `tensorflow-gpu` for faster training
- **Data Quality**: Use high-quality, well-lit face images
- **Preprocessing**: Ensure faces are properly aligned
- **Augmentation**: Balance helps prevent overfitting

## 📈 Model Performance

Expected performance metrics:
- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-90%
- **Test Accuracy**: 80-90%

Factors affecting accuracy:
- Dataset quality and size
- Image preprocessing
- Model architecture
- Training parameters

## 🔧 Customization

### Adding New Face Shapes

1. Add new folders to dataset
2. Update `face_shapes` list in `FaceShapeDetector`
3. Add recommendations in `get_style_recommendations`
4. Retrain the model

### Modifying Recommendations

Edit the `recommendations` dictionary in `get_style_recommendations()` methods.

## 📦 Dependencies

- **TensorFlow 2.13+**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Streamlit**: Web interface
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Visualization
- **PIL**: Image processing
- **scikit-learn**: ML utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset contributors
- TensorFlow team
- OpenCV community
- Streamlit developers

---

**Happy detecting! 🎭✨**
