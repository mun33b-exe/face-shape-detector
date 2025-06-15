"""
Face Detection and Shape Classification Module

This module handles:
1. Face detection using dlib and MediaPipe
2. Facial landmark extraction
3. Face shape classification (oval, round, square, heart, oblong)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib not available. Advanced facial landmark detection will be disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Using OpenCV only for face detection.")

class FaceDetector:
    """Face detection using OpenCV, dlib (optional), and MediaPipe (optional)"""
    
    def __init__(self):
        # Initialize OpenCV face detector (always available)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize dlib face detector and landmark predictor (optional)
        if DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
            
            # You'll need to download this file (see setup instructions)
            self.predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            try:
                self.predictor = dlib.shape_predictor(self.predictor_path)
                self.dlib_predictor_available = True
            except:
                logger.warning("Dlib predictor not found. Please download shape_predictor_68_face_landmarks.dat")
                self.predictor = None
                self.dlib_predictor_available = False
        else:
            self.detector = None
            self.predictor = None
            self.dlib_predictor_available = False
          # Initialize MediaPipe Face Detection (optional)
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        else:
            self.mp_face_detection = None
            self.face_detection = None
    
    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]
      def detect_faces_dlib(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces using dlib (if available)"""
        if not DLIB_AVAILABLE or self.detector is None:
            logger.warning("dlib not available, falling back to OpenCV")
            return self.detect_faces_opencv(image)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.detector(gray)
        return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]
    
    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe (if available)"""
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            logger.warning("MediaPipe not available, falling back to OpenCV")
            opencv_faces = self.detect_faces_opencv(image)
            # Convert OpenCV format to MediaPipe-like format
            return [{'bbox': face, 'confidence': 0.8} for face in opencv_faces]
            
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                faces.append({
                    'bbox': (int(bbox.xmin * w), int(bbox.ymin * h), 
                            int(bbox.width * w), int(bbox.height * h)),
                    'confidence': detection.score[0]
                })
        
        return faces
    
    def get_facial_landmarks(self, image: np.ndarray, face_rect: Tuple) -> Optional[np.ndarray]:
        """Extract 68 facial landmarks using dlib"""
        if self.predictor is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        left, top, right, bottom = face_rect
        rect = dlib.rectangle(left, top, right, bottom)
        
        landmarks = self.predictor(gray, rect)
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        
        return points

class FaceShapeClassifier:
    """Classify face shapes based on facial measurements and ratios"""
    
    def __init__(self):
        self.shape_categories = ['oval', 'round', 'square', 'heart', 'oblong']
    
    def calculate_face_measurements(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate key facial measurements from landmarks"""
        if landmarks is None or len(landmarks) != 68:
            return {}
        
        # Key landmark indices
        # Jawline: 0-16
        # Forehead width: estimated from eyebrow points
        # Face length: top of forehead to bottom of chin
        
        # Face width at different levels
        jaw_width = np.linalg.norm(landmarks[0] - landmarks[16])  # Jawline width
        cheek_width = np.linalg.norm(landmarks[1] - landmarks[15])  # Cheek width
        forehead_width = np.linalg.norm(landmarks[17] - landmarks[26])  # Eyebrow width (approx forehead)
        
        # Face length
        face_length = np.linalg.norm(landmarks[8] - landmarks[27])  # Chin to nose bridge
        
        # Ratios for classification
        face_ratio = face_length / jaw_width if jaw_width > 0 else 0
        jaw_to_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 0
        
        return {
            'face_width': jaw_width,
            'face_length': face_length,
            'forehead_width': forehead_width,
            'jaw_width': jaw_width,
            'cheek_width': cheek_width,
            'face_ratio': face_ratio,
            'jaw_to_forehead_ratio': jaw_to_forehead_ratio
        }
    
    def classify_face_shape(self, measurements: Dict[str, float]) -> str:
        """Classify face shape based on measurements"""
        if not measurements:
            return 'unknown'
        
        face_ratio = measurements.get('face_ratio', 0)
        jaw_to_forehead_ratio = measurements.get('jaw_to_forehead_ratio', 0)
        
        # Classification logic based on facial ratios
        if face_ratio < 1.2:
            if jaw_to_forehead_ratio > 0.8:
                return 'round'
            else:
                return 'square'
        elif face_ratio > 1.5:
            if jaw_to_forehead_ratio < 0.7:
                return 'heart'
            else:
                return 'oblong'
        else:
            return 'oval'  # Most balanced proportions
    
    def analyze_face_shape(self, image: np.ndarray, face_detector: FaceDetector) -> Dict:
        """Complete face shape analysis pipeline"""
        # Detect faces
        faces = face_detector.detect_faces_dlib(image)
        
        if not faces:
            return {'error': 'No faces detected'}
        
        # Use the first (largest) face
        face_rect = faces[0]
        
        # Get landmarks
        landmarks = face_detector.get_facial_landmarks(image, face_rect)
        
        if landmarks is None:
            return {'error': 'Could not extract facial landmarks'}
        
        # Calculate measurements
        measurements = self.calculate_face_measurements(landmarks)
        
        # Classify shape
        face_shape = self.classify_face_shape(measurements)
        
        return {
            'face_shape': face_shape,
            'measurements': measurements,
            'landmarks': landmarks.tolist(),
            'face_rect': face_rect,
            'confidence': 0.8  # Placeholder confidence score
        }

def main():
    """Test the face detection and classification"""
    # Initialize detector and classifier
    detector = FaceDetector()
    classifier = FaceShapeClassifier()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to analyze current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.imshow('Face Shape Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Analyze current frame
            result = classifier.analyze_face_shape(frame, detector)
            print(f"Analysis result: {result}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
