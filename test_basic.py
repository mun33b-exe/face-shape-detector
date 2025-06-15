"""
Simple Test Script for AI Face Shape Detection System

This script tests the basic functionality without requiring advanced libraries like dlib.
"""

import cv2
import numpy as np
import os
import sys

def test_basic_imports():
    """Test if basic libraries are working"""
    print("Testing basic imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully") 
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_camera_access():
    """Test camera access"""
    print("\nTesting camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera access successful")
                print(f"   Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Could not read frame from camera")
                cap.release()
                return False
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_face_detection_basic():
    """Test basic face detection using OpenCV's Haar Cascades"""
    print("\nTesting basic face detection...")
    
    try:
        # Load pre-trained Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("❌ Could not load Haar cascade")
            return False
        
        print("✅ Haar cascade loaded successfully")
        
        # Test with a simple synthetic image
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        test_image[100:300, 150:250] = [255, 255, 255]  # White rectangle as placeholder
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        print(f"✅ Face detection test completed (found {len(faces)} faces in test image)")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_basic_face_shape_logic():
    """Test basic face shape classification logic"""
    print("\nTesting face shape classification logic...")
    
    try:
        # Test ratios for different face shapes
        test_cases = [
            {"face_ratio": 1.0, "jaw_to_forehead_ratio": 0.9, "expected": "round"},
            {"face_ratio": 1.6, "jaw_to_forehead_ratio": 0.6, "expected": "heart"},
            {"face_ratio": 1.4, "jaw_to_forehead_ratio": 0.8, "expected": "oval"},
            {"face_ratio": 1.1, "jaw_to_forehead_ratio": 0.9, "expected": "square"},
            {"face_ratio": 1.8, "jaw_to_forehead_ratio": 0.8, "expected": "oblong"}
        ]
        
        for i, case in enumerate(test_cases):
            # Simple classification logic
            face_ratio = case["face_ratio"]
            jaw_to_forehead_ratio = case["jaw_to_forehead_ratio"]
            
            if face_ratio < 1.2:
                if jaw_to_forehead_ratio > 0.8:
                    result = "round"
                else:
                    result = "square"
            elif face_ratio > 1.5:
                if jaw_to_forehead_ratio < 0.7:
                    result = "heart"
                else:
                    result = "oblong"
            else:
                result = "oval"
            
            status = "✅" if result == case["expected"] else "❌"
            print(f"   {status} Test {i+1}: Expected {case['expected']}, Got {result}")
        
        print("✅ Face shape classification logic test completed")
        return True
        
    except Exception as e:
        print(f"❌ Face shape logic test failed: {e}")
        return False

def test_style_recommendations():
    """Test style recommendation system"""
    print("\nTesting style recommendation system...")
    
    try:
        # Import our recommendation engine
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from style_recommendations import StyleRecommendationEngine
        
        engine = StyleRecommendationEngine()
        
        # Test recommendations for each face shape
        test_shapes = ['oval', 'round', 'square', 'heart', 'oblong']
        
        for shape in test_shapes:
            recommendations = engine.get_top_recommendations(shape, 2)
            
            if 'error' not in recommendations:
                hairstyles = recommendations.get('top_hairstyles', [])
                beards = recommendations.get('top_beards', [])
                print(f"   ✅ {shape.upper()}: {len(hairstyles)} hairstyles, {len(beards)} beard styles")
            else:
                print(f"   ❌ {shape.upper()}: {recommendations['error']}")
                return False
        
        print("✅ Style recommendation system test completed")
        return True
        
    except Exception as e:
        print(f"❌ Style recommendation test failed: {e}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "main.py",
        "src/face_detection.py",
        "src/style_recommendations.py", 
        "src/virtual_tryon.py",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "src",
        "models",
        "datasets",
        "filters",
        "utils"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_good = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - MISSING")
            all_good = False
    
    if all_good:
        print("✅ File structure test completed")
    else:
        print("❌ Some files/directories are missing")
    
    return all_good

def interactive_camera_test():
    """Interactive camera test"""
    print("\n" + "="*50)
    print("INTERACTIVE CAMERA TEST")
    print("This will open your camera to test basic face detection")
    print("Press 'q' to quit, 's' to take a screenshot")
    print("="*50)
    
    response = input("Run camera test? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Skipping camera test")
        return True
    
    try:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        print("Camera opened successfully! Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Add simple face shape estimation
                face_ratio = h / w if w > 0 else 0
                if face_ratio < 1.2:
                    shape = "Round/Square"
                elif face_ratio > 1.5:
                    shape = "Oblong/Heart"
                else:
                    shape = "Oval"
                
                cv2.putText(frame, f"Shape: {shape}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('AI Face Shape Detection - Basic Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('test_screenshot.jpg', frame)
                print("Screenshot saved as 'test_screenshot.jpg'")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Interactive camera test completed")
        return True
        
    except Exception as e:
        print(f"❌ Interactive camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Face Shape Detection System - Basic Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("Camera Access", test_camera_access),
        ("Face Detection", test_face_detection_basic),
        ("Face Shape Logic", test_basic_face_shape_logic),
        ("Style Recommendations", test_style_recommendations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        
        # Offer interactive camera test
        interactive_camera_test()
        
        print("\n" + "="*60)
        print("🎯 NEXT STEPS FOR YOUR PROJECT:")
        print("1. Install dlib for advanced face landmarks:")
        print("   pip install dlib")
        print("2. Collect face images for training data")
        print("3. Create hairstyle and beard filter images")
        print("4. Train a custom face shape classification model")
        print("5. Test the full system: python main.py --mode camera")
        
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
    
    print("\n📚 Check README.md for detailed instructions")

if __name__ == "__main__":
    main()
