# AI Face Shape Detection & Styling Recommendation System

An AI-powered system that detects face shapes from images or camera feed and recommends suitable hairstyles and beard styles with virtual try-on capabilities.

## 🎯 Project Overview

This system combines computer vision, machine learning, and image processing to:

1. **Detect faces** using OpenCV and dlib
2. **Classify face shapes** into 5 categories (oval, round, square, heart, oblong)
3. **Recommend hairstyles and beard styles** based on face shape
4. **Apply virtual filters** to preview how styles would look

## 🏗️ Project Structure

```
AI Project/
├── src/                          # Source code
│   ├── face_detection.py         # Face detection & shape classification
│   ├── style_recommendations.py  # Style recommendation engine
│   └── virtual_tryon.py          # Virtual try-on filters
├── models/                       # Pre-trained models
│   └── shape_predictor_68_face_landmarks.dat
├── datasets/                     # Training data
│   ├── train/                    # Training images
│   ├── validation/               # Validation images
│   ├── test/                     # Test images
│   └── annotations/              # Image annotations
├── filters/                      # Style filter images
│   ├── hairstyles/               # Hairstyle overlays
│   └── beards/                   # Beard overlays
├── utils/                        # Utility functions
│   └── dataset_utils.py          # Dataset management
├── main.py                       # Main application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- Windows/macOS/Linux

### Installation

1. **Clone or download this project** to your local machine

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models:**
   ```bash
   python utils/dataset_utils.py
   ```
   This will automatically download the dlib face landmark predictor.

4. **Run the application:**
   
   **Camera Mode (Real-time):**
   ```bash
   python main.py --mode camera
   ```
   
   **Image Mode (Single image):**
   ```bash
   python main.py --mode image --image path/to/your/image.jpg
   ```

## 🎮 How to Use

### Camera Mode Controls
- **'s'** - Analyze current frame for face shape
- **'1-3'** - Apply hairstyle filters (after analysis)
- **'4-6'** - Apply beard filters (after analysis)  
- **'r'** - Reset to original view
- **'c'** - Capture and save current image
- **'q'** - Quit application

### Image Mode Controls
- **Numbers 1-6** - Apply different style filters
- **'r'** - Reset to original image
- **'q'** - Quit and close windows

## 🧠 How It Works

### 1. Face Detection
- Uses **dlib** for robust face detection
- Extracts **68 facial landmarks** for precise measurements
- Fallback to **MediaPipe** for additional detection capabilities

### 2. Face Shape Classification
The system analyzes facial proportions to classify faces into:

- **Oval**: Balanced proportions, face length 1.5x face width
- **Round**: Face width and length are nearly equal
- **Square**: Angular jawline, face width similar to face length  
- **Heart**: Wide forehead, narrow chin
- **Oblong**: Face length significantly longer than width

### 3. Style Recommendations
Based on established styling principles:

#### For Oval Faces (Most versatile):
- Classic side part, textured quiff, slicked back
- Full beard, goatee, stubble

#### For Round Faces (Add height/angles):
- High fade, pompadour, angular fringe
- Chin strap, soul patch

#### For Square Faces (Soften angles):
- Textured crop, messy waves, long layers
- Rounded beard, circle beard

#### For Heart Faces (Balance forehead):
- Side swept bangs, chin-length bob, low fade
- Full chin beard, extended goatee

#### For Oblong Faces (Add width):
- Fringe/bangs, layered cut, wide caesar
- Horizontal mustache, wide beard

### 4. Virtual Try-On
- **Image overlay** technology aligns filters with facial landmarks
- **Alpha blending** for realistic transparency effects
- **Real-time processing** for immediate feedback

## 📚 What You Need to Learn

### Core Concepts

1. **Computer Vision Basics**
   - Image processing with OpenCV
   - Face detection algorithms
   - Facial landmark detection

2. **Machine Learning**
   - Feature extraction from images
   - Classification algorithms
   - Model training and validation

3. **Python Libraries**
   - **OpenCV**: Image processing and computer vision
   - **dlib**: Face detection and landmark extraction
   - **NumPy**: Numerical computations
   - **PIL/Pillow**: Image manipulation
   - **MediaPipe**: Google's ML framework for perception tasks

### Implementation Steps

1. **Set up development environment**
2. **Understand face detection basics**
3. **Learn facial landmark extraction**
4. **Implement face shape classification logic**
5. **Create recommendation database**
6. **Develop image overlay techniques**
7. **Build user interface**
8. **Test and refine the system**

## 📊 Dataset Requirements

### Face Shape Classification Dataset

You'll need labeled images for each face shape category:

- **Training set**: ~1000-5000 images per category
- **Validation set**: ~200-500 images per category  
- **Test set**: ~200-500 images per category

### Recommended Datasets

1. **CelebA**: Celebrity faces with annotations
2. **FFHQ**: High-quality face images
3. **Custom collection**: Gather images with proper permissions

### Filter Images

Create or collect PNG images with transparent backgrounds:

- **Hairstyles**: Various cuts for each face shape
- **Beards**: Different beard styles and lengths
- **Format**: 512x512 pixels, PNG with alpha channel

## 🛠️ Development Tips

### 1. Start Simple
- Begin with basic face detection
- Test with your own photos first
- Add complexity gradually

### 2. Handle Edge Cases
- Multiple faces in image
- Poor lighting conditions
- Side profiles or tilted faces
- Glasses, masks, or obstructions

### 3. Improve Accuracy
- Collect diverse training data
- Fine-tune classification thresholds
- Add user feedback mechanisms
- Implement ensemble methods

### 4. Performance Optimization
- Resize images for faster processing
- Use appropriate data types
- Implement caching for repeated operations
- Consider GPU acceleration for real-time use

## 🎨 Customization Options

### Adding New Face Shapes
1. Update classification logic in `face_detection.py`
2. Add recommendations in `style_recommendations.py`
3. Create corresponding filter images

### Adding New Styles
1. Create filter images in appropriate directories
2. Update recommendation database
3. Test alignment with different face shapes

### UI Improvements  
- Add web interface using Streamlit or Flask
- Implement mobile app version
- Create desktop GUI with tkinter or PyQt

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Camera not working**: Check camera permissions and drivers
3. **dlib model not found**: Run the dataset utility to download models
4. **Poor detection accuracy**: Ensure good lighting and face visibility
5. **Filter alignment issues**: Adjust landmark-based positioning

### Performance Issues

- **Slow processing**: Reduce image resolution
- **High memory usage**: Process images in batches
- **Real-time lag**: Optimize code and consider frame skipping

## 📈 Future Enhancements

### Technical Improvements
- Deep learning-based face shape classification
- More sophisticated filter blending
- 3D face modeling and projection
- Real-time hair color modification

### Feature Additions
- Makeup recommendations
- Eyewear suggestions
- Age progression/regression
- Style trend analysis
- Social sharing capabilities

### Platform Extensions
- Mobile app development
- Web-based version
- API for third-party integration
- Cloud-based processing

## 📝 Assignment Submission

### What to Include

1. **Source code** with comments
2. **Demonstration video** showing the system working
3. **Technical documentation** explaining your approach
4. **Test results** with various face shapes
5. **Challenges faced** and how you solved them

### Evaluation Criteria

- **Functionality**: Does the system work as intended?
- **Accuracy**: How well does it classify face shapes?
- **User Experience**: Is it easy to use?
- **Code Quality**: Is the code well-structured and documented?
- **Innovation**: Any creative improvements or features?

## 📖 Learning Resources

### Tutorials and Documentation
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [dlib Face Detection](http://dlib.net/face_detector.py.html)
- [MediaPipe Face Detection](https://mediapipe.dev/solutions/face_detection)

### Books
- "Learning OpenCV" by Gary Bradski
- "Computer Vision: Algorithms and Applications" by Richard Szeliski
- "Hands-On Machine Learning" by Aurélien Géron

### Online Courses
- Computer Vision courses on Coursera/edX
- Deep Learning Specialization
- Python for Computer Vision with OpenCV

## 🤝 Contributing

Feel free to improve this project by:
- Adding new face shape categories
- Improving classification accuracy
- Creating better filter images
- Optimizing performance
- Adding new features

## 📄 License

This project is for educational purposes. Please respect copyright and privacy when using face images and ensure compliance with relevant regulations.

---

**Good luck with your AI project! 🚀**

For questions or issues, please refer to the troubleshooting section or consult the documentation of the individual libraries used.
