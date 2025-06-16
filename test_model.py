"""
Test script for face shape detection
Use this to test your trained model on new images
"""

from face_shape_detector import FaceShapeDetector
from camera_detector import CameraFaceShapeDetector
import os

def test_single_image():
    """Test the model on a single image"""
    print("🔍 Testing Face Shape Detection on Image")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('face_shape_model.h5'):
        print("❌ Model not found. Please train the model first using train_model.py")
        return
    
    # Get image path from user
    image_path = input("Enter the path to your image: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ Image not found. Please check the path.")
        return
    
    try:
        # Initialize detector
        detector = CameraFaceShapeDetector()
        
        # Detect face shape
        results = detector.detect_from_image(image_path)
        
        if results:
            print("\n🎯 Detection Results:")
            print("=" * 30)
            
            for result in results:
                print(f"\n👤 Face {result['face_number']}:")
                print(f"   Face Shape: {result['face_shape']}")
                print(f"   Confidence: {result['confidence']:.2%}")
                print(f"   Hair Recommendations: {result['recommendations']['hair']}")
                print(f"   Beard Recommendations: {result['recommendations']['beard']}")
            
            print(f"\n📁 Result saved as: result_{os.path.basename(image_path)}")
        else:
            print("❌ No faces detected in the image.")
            
    except Exception as e:
        print(f"❌ Error during detection: {e}")

def test_camera():
    """Test the model with camera"""
    print("📹 Testing Face Shape Detection with Camera")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('face_shape_model.h5'):
        print("❌ Model not found. Please train the model first using train_model.py")
        return
    
    try:
        # Initialize detector
        detector = CameraFaceShapeDetector()
        
        print("🎥 Starting camera detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Start camera detection
        detector.detect_from_camera()
        
    except Exception as e:
        print(f"❌ Error during camera detection: {e}")

def main():
    """Main menu"""
    print("🎭 Face Shape Detection Test Suite")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. 📸 Test with image file")
        print("2. 📹 Test with camera")
        print("3. ❌ Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            test_single_image()
        elif choice == '2':
            test_camera()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
