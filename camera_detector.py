import cv2
import numpy as np
import tensorflow as tf
from face_shape_detector import FaceShapeDetector

class FaceDetector:
    def __init__(self):
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image):
        """Detect faces in image using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_face(self, image, face_coords):
        """Extract face region from image"""
        x, y, w, h = face_coords
        # Add some padding
        padding = int(0.2 * min(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_img = image[y1:y2, x1:x2]
        return face_img

class CameraFaceShapeDetector:
    def __init__(self, model_path='face_shape_model.h5'):
        self.face_detector = FaceDetector()
        self.shape_detector = FaceShapeDetector()
        
        # Load pre-trained model
        try:
            self.shape_detector.load_model(model_path)
            print("Model loaded successfully!")
        except:
            print("Model not found. Please train the model first.")
            return
    
    def detect_from_camera(self):
        """Real-time face shape detection from camera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit, 's' to save current prediction")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face
                face_img = self.face_detector.extract_face(frame, (x, y, w, h))
                
                if face_img.size > 0:
                    # Convert BGR to RGB for prediction
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    # Predict face shape
                    try:
                        face_shape, confidence = self.shape_detector.predict_face_shape(face_rgb)
                        
                        # Display prediction
                        text = f"{face_shape}: {confidence:.2f}"
                        cv2.putText(frame, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Get style recommendations
                        recommendations = self.get_style_recommendations(face_shape)
                        
                        # Display recommendations
                        y_offset = y + h + 30
                        cv2.putText(frame, "Recommended:", (x, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"Hair: {recommendations['hair']}", (x, y_offset + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(frame, f"Beard: {recommendations['beard']}", (x, y_offset + 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
            
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
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image")
            return
        
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face
            face_img = self.face_detector.extract_face(image, (x, y, w, h))
            
            if face_img.size > 0:
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
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, f"{face_shape}: {confidence:.2f}", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save result image
        cv2.imwrite('result_' + image_path.split('/')[-1], image)
        
        return results
    
    def get_style_recommendations(self, face_shape):
        """Get hairstyle and beard recommendations based on face shape"""
        recommendations = {
            'Heart': {
                'hair': ['Side-swept bangs', 'Long layers', 'Chin-length bob'],
                'beard': ['Light stubble', 'Goatee', 'Soul patch']
            },
            'Oval': {
                'hair': ['Most styles work', 'Shoulder-length', 'Pixie cut', 'Long waves'],
                'beard': ['Full beard', 'Stubble', 'Mustache', 'Clean shaven']
            },
            'Round': {
                'hair': ['Layered cuts', 'Side part', 'Angular styles', 'Long straight'],
                'beard': ['Angular beard', 'Goatee', 'Extended goatee']
            },
            'Square': {
                'hair': ['Soft layers', 'Side-swept', 'Wavy styles', 'Rounded cuts'],
                'beard': ['Rounded beard', 'Circle beard', 'Light stubble']
            },
            'Oblong': {
                'hair': ['Bangs', 'Layered bob', 'Wide styles', 'Curly/wavy'],
                'beard': ['Full beard', 'Mutton chops', 'Wide mustache']
            }
        }
        
        if face_shape in recommendations:
            styles = recommendations[face_shape]
            return {
                'hair': ', '.join(styles['hair'][:2]),  # Show top 2 recommendations
                'beard': ', '.join(styles['beard'][:2])
            }
        else:
            return {'hair': 'Consult a stylist', 'beard': 'Consult a stylist'}

if __name__ == "__main__":
    # Initialize camera detector
    detector = CameraFaceShapeDetector()
    
    print("Choose an option:")
    print("1. Real-time camera detection")
    print("2. Detect from image file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        detector.detect_from_camera()
    elif choice == '2':
        image_path = input("Enter image path: ")
        results = detector.detect_from_image(image_path)
        
        if results:
            for result in results:
                print(f"\nFace {result['face_number']}:")
                print(f"Face Shape: {result['face_shape']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Hair Recommendations: {result['recommendations']['hair']}")
                print(f"Beard Recommendations: {result['recommendations']['beard']}")
    else:
        print("Invalid choice")
