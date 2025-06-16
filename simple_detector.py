"""
Simple camera interface for the lightweight face shape detector
Works with basic OpenCV and scikit-learn
"""

import cv2
import numpy as np
from lightweight_detector import LightweightFaceShapeDetector
import os

class SimpleCameraDetector:
    def __init__(self, model_path='lightweight_face_shape_model.pkl'):
        self.face_cascade = None
        self.shape_detector = LightweightFaceShapeDetector()
        self.model_loaded = False
        
        # Load face detection cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("Face detection cascade loaded successfully")
        except Exception as e:
            print(f"Error loading face cascade: {e}")
        
        # Load face shape model
        if os.path.exists(model_path):
            self.model_loaded = self.shape_detector.load_model(model_path)
        else:
            print(f"Model not found: {model_path}")
            print("Please train the model first using lightweight_detector.py")
    
    def detect_faces(self, image):
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def extract_face(self, image, face_coords):
        """Extract face region from image"""
        x, y, w, h = face_coords
        # Add padding
        padding = int(0.1 * min(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_img = image[y1:y2, x1:x2]
        return face_img
    
    def get_style_recommendations(self, face_shape):
        """Get style recommendations based on face shape"""
        recommendations = {
            'Heart': {
                'hair': ['Side-swept bangs', 'Long layers'],
                'beard': ['Light stubble', 'Goatee']
            },
            'Oval': {
                'hair': ['Most styles work', 'Shoulder-length'],
                'beard': ['Full beard', 'Stubble']
            },
            'Round': {
                'hair': ['Layered cuts', 'Side part'],
                'beard': ['Angular beard', 'Goatee']
            },
            'Square': {
                'hair': ['Soft layers', 'Side-swept'],
                'beard': ['Rounded beard', 'Circle beard']
            },
            'Oblong': {
                'hair': ['Bangs', 'Layered bob'],
                'beard': ['Full beard', 'Wide mustache']
            }
        }
        
        if face_shape in recommendations:
            return recommendations[face_shape]
        else:
            return {'hair': ['Consult stylist'], 'beard': ['Consult stylist']}
    
    def detect_from_camera(self):
        """Real-time face shape detection from camera"""
        if not self.model_loaded:
            print("Model not loaded. Cannot proceed.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera started. Press 'q' to quit, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face
                face_img = self.extract_face(frame, (x, y, w, h))
                
                if face_img.size > 0:
                    try:
                        # Convert BGR to RGB for prediction
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        # Predict face shape
                        face_shape, confidence = self.shape_detector.predict_face_shape(face_rgb)
                        
                        # Display prediction
                        text = f"{face_shape}: {confidence:.2f}"
                        cv2.putText(frame, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Get recommendations
                        recommendations = self.get_style_recommendations(face_shape)
                        
                        # Display recommendations
                        y_offset = y + h + 25
                        cv2.putText(frame, f"Hair: {recommendations['hair'][0]}", 
                                  (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"Beard: {recommendations['beard'][0]}", 
                                  (x, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        cv2.putText(frame, "Prediction Error", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Face Shape Detection', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('detected_face.jpg', frame)
                print("Frame saved as detected_face.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_image(self, image_path):
        """Detect face shape from image file"""
        if not self.model_loaded:
            print("Model not loaded. Cannot proceed.")
            return []
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []
        
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image")
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return []
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face
            face_img = self.extract_face(image, (x, y, w, h))
            
            if face_img.size > 0:
                try:
                    # Convert BGR to RGB for prediction
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    # Predict face shape
                    face_shape, confidence = self.shape_detector.predict_face_shape(face_rgb)
                    
                    # Get style recommendations
                    recommendations = self.get_style_recommendations(face_shape)
                    
                    result = {
                        'face_number': i + 1,
                        'face_shape': face_shape,
                        'confidence': confidence,
                        'recommendations': recommendations
                    }
                    results.append(result)
                    
                    # Draw on image
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image, f"{face_shape}: {confidence:.2f}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error processing face {i+1}: {e}")
        
        # Save result image
        result_path = f'result_{os.path.basename(image_path)}'
        cv2.imwrite(result_path, image)
        print(f"Result saved as: {result_path}")
        
        return results

def main():
    """Main interface"""
    print("🎭 Simple Face Shape Detection")
    print("=" * 40)
    
    detector = SimpleCameraDetector()
    
    while True:
        print("\nChoose an option:")
        print("1. 📸 Detect from image file")
        print("2. 📹 Real-time camera detection")
        print("3. ❌ Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip().strip('"')
            results = detector.detect_from_image(image_path)
            
            if results:
                print("\n🎯 Detection Results:")
                for result in results:
                    print(f"\n👤 Face {result['face_number']}:")
                    print(f"  Shape: {result['face_shape']}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Hair: {', '.join(result['recommendations']['hair'])}")
                    print(f"  Beard: {', '.join(result['recommendations']['beard'])}")
        
        elif choice == '2':
            detector.detect_from_camera()
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
