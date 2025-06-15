"""
Setup Script for AI Face Shape Detection System

This script automates the initial setup process including:
- Installing dependencies
- Downloading required models
- Creating sample data
- Validating the installation
"""

import os
import sys
import subprocess
import urllib.request
import bz2
from pathlib import Path

def install_requirements():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_dlib_model():
    """Download dlib face landmark predictor model"""
    print("Downloading dlib face landmarks model...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "shape_predictor_68_face_landmarks.dat"
    
    if model_path.exists():
        print("✅ Model already exists!")
        return True
    
    model_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = model_path.with_suffix(".dat.bz2")
    
    try:
        print("Downloading... (this may take a few minutes)")
        urllib.request.urlretrieve(model_url, compressed_path)
        
        print("Extracting...")
        with bz2.BZ2File(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Clean up compressed file
        compressed_path.unlink()
        
        print("✅ Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Please manually download from: https://github.com/davisking/dlib-models")
        return False

def create_directory_structure():
    """Create required directory structure"""
    print("Creating directory structure...")
    
    directories = [
        "models",
        "datasets/train/oval",
        "datasets/train/round", 
        "datasets/train/square",
        "datasets/train/heart",
        "datasets/train/oblong",
        "datasets/validation/oval",
        "datasets/validation/round",
        "datasets/validation/square", 
        "datasets/validation/heart",
        "datasets/validation/oblong",
        "datasets/test/oval",
        "datasets/test/round",
        "datasets/test/square",
        "datasets/test/heart", 
        "datasets/test/oblong",
        "datasets/annotations",
        "filters/hairstyles",
        "filters/beards"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def create_sample_filters():
    """Create sample filter images for testing"""
    print("Creating sample filter images...")
    
    try:
        import cv2
        import numpy as np
        
        # Create sample hairstyle filters
        hairstyles_dir = Path("filters/hairstyles")
        for i, color in enumerate([(101, 67, 33), (139, 69, 19), (62, 39, 35)]):
            img = np.zeros((200, 300, 4), dtype=np.uint8)
            img[:, :, :3] = color  # BGR
            img[:, :, 3] = 180     # Alpha
            cv2.imwrite(str(hairstyles_dir / f"sample_hair_{i+1}.png"), img)
        
        # Create sample beard filters  
        beards_dir = Path("filters/beards")
        for i, color in enumerate([(101, 67, 33), (139, 69, 19), (160, 82, 45)]):
            img = np.zeros((150, 200, 4), dtype=np.uint8)
            img[:, :, :3] = color  # BGR
            img[:, :, 3] = 200     # Alpha
            cv2.imwrite(str(beards_dir / f"sample_beard_{i+1}.png"), img)
        
        print("✅ Sample filters created!")
        return True
        
    except ImportError:
        print("⚠️  OpenCV not available, skipping sample filter creation")
        return False

def validate_installation():
    """Validate that everything is set up correctly"""
    print("Validating installation...")
    
    checks = []
    
    # Check if required files exist
    required_files = [
        "requirements.txt",
        "main.py", 
        "src/face_detection.py",
        "src/style_recommendations.py",
        "src/virtual_tryon.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            checks.append(f"✅ {file_path}")
        else:
            checks.append(f"❌ {file_path} - MISSING")
    
    # Check if model exists
    model_path = Path("models/shape_predictor_68_face_landmarks.dat")
    if model_path.exists():
        checks.append(f"✅ {model_path}")
    else:
        checks.append(f"❌ {model_path} - MISSING")
    
    # Check if key directories exist
    key_dirs = ["datasets", "filters", "models", "src"]
    for directory in key_dirs:
        if Path(directory).exists():
            checks.append(f"✅ {directory}/")
        else:
            checks.append(f"❌ {directory}/ - MISSING")
    
    print("\nValidation Results:")
    for check in checks:
        print(f"  {check}")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import cv2
        print("  ✅ OpenCV")
    except ImportError:
        print("  ❌ OpenCV - Run: pip install opencv-python")
    
    try:
        import numpy
        print("  ✅ NumPy")
    except ImportError:
        print("  ❌ NumPy - Run: pip install numpy")
    
    try:
        import dlib
        print("  ✅ dlib")
    except ImportError:
        print("  ❌ dlib - Run: pip install dlib")
    
    try:
        import mediapipe
        print("  ✅ MediaPipe")
    except ImportError:
        print("  ❌ MediaPipe - Run: pip install mediapipe")

def main():
    """Main setup function"""
    print("🚀 AI Face Shape Detection System Setup")
    print("=" * 50)
    
    steps = [
        ("Creating directory structure", create_directory_structure),
        ("Installing dependencies", install_requirements),
        ("Downloading dlib model", download_dlib_model),
        ("Creating sample filters", create_sample_filters),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        step_func()
    
    print(f"\n🔍 Validating installation...")
    validate_installation()
    
    print(f"\n{'='*50}")
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Test the system: python main.py --mode camera")
    print("2. Try with an image: python main.py --mode image --image your_photo.jpg")
    print("3. Read README.md for detailed instructions")
    print("4. Replace sample filters with real hairstyle/beard images")
    print("\n💡 For your project:")
    print("- Collect face images for each shape category")
    print("- Create/find hairstyle and beard filter images")
    print("- Train a custom face shape classification model")

if __name__ == "__main__":
    main()
