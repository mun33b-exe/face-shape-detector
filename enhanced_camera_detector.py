"""
Enhanced Camera Detector for Advanced Face Shape Detection
Optimized for GTX 1660 Ti with real-time performance
Uses the advanced CNN model for high accuracy predictions
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from datetime import datetime
import time

# Import the advanced face detector
try:
    from advanced_face_detector_v2 import AdvancedFaceShapeDetector
except ImportError:
    print("⚠️ Could not import AdvancedFaceShapeDetector. Using standalone mode.")

class EnhancedCameraDetector:
    def __init__(self, model_path='advanced_face_shape_model.h5'):
        self.model = None
        self.face_shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.img_size = (224, 224)
        self.model_loaded = False
        
        print("🎥 Enhanced Camera Face Shape Detector")
        print("🚀 Optimized for GTX 1660 Ti")
        
        # Enhanced style recommendations
        self.style_recommendations = {
            'Heart': {
                'hairstyles': [
                    '✨ Long layers to balance forehead',
                    '🌊 Side-swept bangs',
                    '💇 Chin-length bob',
                    '🎀 Beach waves'
                ],
                'beard_styles': [
                    '🧔 Light stubble',
                    '🎯 Full goatee',
                    '⭕ Circle beard',
                    '🔺 Van Dyke'
                ],
                'avoid': [
                    '❌ Center parts',
                    '❌ Heavy blunt bangs'
                ]
            },
            'Oblong': {
                'hairstyles': [
                    '📏 Layered cuts with width',
                    '🔄 Side parts',
                    '✂️ Textured crop',
                    '🌊 Wavy styles'
                ],
                'beard_styles': [
                    '🧔 Full beard',
                    '📐 Mutton chops',
                    '🎯 Extended goatee',
                    '🔄 Horseshoe mustache'
                ],
                'avoid': [
                    '❌ Very long straight styles',
                    '❌ Center parts'
                ]
            },
            'Oval': {
                'hairstyles': [
                    '🌟 Most styles work perfectly!',
                    '✂️ Pixie cut',
                    '🌊 Long waves',
                    '💼 Slicked back'
                ],
                'beard_styles': [
                    '😊 Any style works!',
                    '🧔 Full beard',
                    '✨ Clean shaven',
                    '🎯 Light stubble'
                ],
                'avoid': [
                    '⚠️ Very few restrictions!'
                ]
            },
            'Round': {
                'hairstyles': [
                    '📐 Angular cuts',
                    '⬆️ High fade',
                    '💫 Pompadour',
                    '📏 Side parts'
                ],
                'beard_styles': [
                    '⚓ Anchor beard',
                    '🎯 Soul patch',
                    '📏 Chin strap',
                    '👑 Imperial'
                ],
                'avoid': [
                    '❌ Round cuts',
                    '❌ Center parts'
                ]
            },
            'Square': {
                'hairstyles': [
                    '🌊 Soft layers',
                    '✨ Textured quiff',
                    '📐 Side-swept',
                    '🌀 Curly styles'
                ],
                'beard_styles': [
                    '⭕ Circle beard',
                    '🎯 Balbo',
                    '🔄 Rounded goatee',
                    '✨ Light stubble'
                ],
                'avoid': [
                    '❌ Angular cuts',
                    '❌ Sharp beard lines'
                ]
            }
        }
        
        # Load face detection
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
            print("✅ Face detection loaded")
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return
        
        # Load the advanced model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the advanced deep learning model"""
        try:
            if os.path.exists(model_path):
                print(f"📊 Loading model: {model_path}")
                self.model = keras.models.load_model(model_path)
                self.model_loaded = True
                
                # Load metadata if available
                metadata_path = model_path.replace('.h5', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.face_shapes = metadata.get('dataset_info', {}).get('face_shapes', self.face_shapes)
                    self.img_size = tuple(metadata.get('dataset_info', {}).get('img_size', self.img_size))
                    
                    print(f"✅ Model loaded: {metadata.get('model_info', {}).get('type', 'Advanced CNN')}")
                    print(f"🎯 Face shapes: {self.face_shapes}")
                    print(f"📏 Input size: {self.img_size}")
                else:
                    print("✅ Model loaded (no metadata)")
                
                return True
            else:
                print(f"❌ Model not found: {model_path}")
                print("💡 Train the model first using: python advanced_face_detector_v2.py")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_face(self, face_img):
        """Preprocess face for the advanced model"""
        try:
            # Resize with high-quality interpolation
            face_resized = cv2.resize(face_img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0,1]
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict_face_shape(self, face_img):
        """Predict face shape with confidence and probabilities"""
        if not self.model_loaded:
            return "No Model", 0.0, {}
        
        try:
            # Preprocess
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Error", 0.0, {}
            
            # Predict
            predictions = self.model.predict(processed_face, verbose=0)
            
            # Get results
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # All probabilities
            all_probs = {}
            for i, shape in enumerate(self.face_shapes):
                all_probs[shape] = float(predictions[0][i])
            
            face_shape = self.face_shapes[predicted_class]
            
            return face_shape, float(confidence), all_probs
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, {}
    
    def detect_faces(self, frame):
        """Detect faces with optimized parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale detection for better accuracy
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def process_frame(self, frame):
        """Process a single frame and return results"""
        results = []
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face with padding
            padding = int(0.2 * min(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                # Predict face shape
                shape, confidence, probabilities = self.predict_face_shape(face_img)
                
                results.append({
                    'bbox': (x, y, w, h),
                    'face_shape': shape,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
        
        return results
    
    def draw_enhanced_results(self, frame, results):
        """Draw enhanced results with better visualization"""
        for result in results:
            x, y, w, h = result['bbox']
            shape = result['face_shape']
            confidence = result['confidence']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - High confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - Medium confidence
            else:
                color = (0, 165, 255)  # Orange - Low confidence
            
            # Face rectangle with thickness based on confidence
            thickness = int(2 + confidence * 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Enhanced label with background
            label = f"{shape}: {confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Background rectangle for text
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Text
            cv2.putText(frame, label, (x, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_style_panel(self, frame, results):
        """Draw enhanced style recommendations panel"""
        if not results:
            return frame
        
        # Get best result
        best_result = max(results, key=lambda x: x['confidence'])
        shape = best_result['face_shape']
        confidence = best_result['confidence']
        
        if confidence < 0.5 or shape not in self.style_recommendations:
            return frame
        
        recommendations = self.style_recommendations[shape]
        
        # Panel settings
        panel_width = 420
        panel_height = 250
        margin = 10
        start_x = frame.shape[1] - panel_width - margin
        start_y = margin
        
        # Background with transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), 
                     (start_x + panel_width, start_y + panel_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Border
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + panel_width, start_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        title = f"Style Guide: {shape} Face"
        cv2.putText(frame, title, (start_x + 15, start_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_pos = start_y + 60
        
        # Hairstyles
        cv2.putText(frame, "HAIRSTYLES:", (start_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 25
        
        for style in recommendations['hairstyles'][:3]:
            cv2.putText(frame, style, (start_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20
        
        y_pos += 10
        
        # Beard styles
        cv2.putText(frame, "BEARD STYLES:", (start_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 25
        
        for style in recommendations['beard_styles'][:3]:
            cv2.putText(frame, style, (start_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20
        
        y_pos += 10
        
        # What to avoid
        if recommendations['avoid']:
            cv2.putText(frame, "AVOID:", (start_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_pos += 25
            
            for avoid in recommendations['avoid'][:2]:
                cv2.putText(frame, avoid, (start_x + 20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_pos += 20
        
        return frame
    
    def draw_probability_bars(self, frame, results):
        """Draw probability visualization"""
        if not results:
            return frame
        
        best_result = max(results, key=lambda x: x['confidence'])
        probabilities = best_result['probabilities']
        
        # Panel settings
        panel_width = 320
        panel_height = 180
        start_x = 10
        start_y = frame.shape[0] - panel_height - 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), 
                     (start_x + panel_width, start_y + panel_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Border
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + panel_width, start_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "PROBABILITIES", (start_x + 15, start_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Bars
        y_offset = 50
        bar_width = 180
        bar_height = 20
        
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (shape, prob) in enumerate(sorted_probs):
            y_pos = start_y + y_offset + i * 25
            
            # Shape name
            cv2.putText(frame, f"{shape}:", (start_x + 15, y_pos + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Bar background
            cv2.rectangle(frame, (start_x + 80, y_pos), 
                         (start_x + 80 + bar_width, y_pos + bar_height), 
                         (50, 50, 50), -1)
            
            # Bar fill
            fill_width = int(bar_width * prob)
            if i == 0:  # Highest probability
                bar_color = (0, 255, 0)
            elif prob > 0.3:
                bar_color = (0, 255, 255)
            else:
                bar_color = (255, 255, 255)
            
            cv2.rectangle(frame, (start_x + 80, y_pos), 
                         (start_x + 80 + fill_width, y_pos + bar_height), 
                         bar_color, -1)
            
            # Percentage
            cv2.putText(frame, f"{prob:.1%}", (start_x + 270, y_pos + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run_camera(self, camera_id=0, show_style=True, show_probs=True):
        """Run enhanced real-time detection"""
        if not self.model_loaded:
            print("❌ No model loaded. Train the model first!")
            return
        
        print("🎥 Starting Enhanced Camera Detection")
        print("🎮 Optimized for GTX 1660 Ti")
        print("\n📋 Controls:")
        print("  's' - Save screenshot")
        print("  'r' - Toggle style recommendations")
        print("  'p' - Toggle probability bars")
        print("  'q' or ESC - Quit")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Camera settings for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance tracking
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror for better UX
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw results
            frame = self.draw_enhanced_results(frame, results)
            
            if show_style:
                frame = self.draw_style_panel(frame, results)
            
            if show_probs:
                frame = self.draw_probability_bars(frame, results)
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Status info
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if tf.config.list_physical_devices('GPU'):
                cv2.putText(frame, "GPU: GTX 1660 Ti", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Title
            cv2.putText(frame, "Enhanced Face Shape Detection", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Enhanced Face Shape Detection - GTX 1660 Ti', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'enhanced_detection_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                show_style = not show_style
                print(f"📋 Style panel: {'ON' if show_style else 'OFF'}")
            elif key == ord('p'):
                show_probs = not show_probs
                print(f"📊 Probability bars: {'ON' if show_probs else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Enhanced detection stopped")

def main():
    print("🎭 Enhanced Face Shape Detection")
    print("🎮 GTX 1660 Ti Optimized")
    print("=" * 50)
    
    detector = EnhancedCameraDetector()
    
    if detector.model_loaded:
        detector.run_camera()
    else:
        print("\n💡 To get started:")
        print("1. Run: python advanced_face_detector_v2.py")
        print("2. Wait for training to complete")
        print("3. Run this camera detector again")

if __name__ == "__main__":
    main()
