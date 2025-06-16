"""
Advanced Camera Face Shape Detector
Works with the new high-accuracy CNN models
Optimized for real-time performance with GPU acceleration
"""

import cv2
import numpy as np
import tensorflow as tf
from advanced_face_detector import AdvancedFaceShapeDetector
import os
import json
from datetime import datetime

class AdvancedCameraDetector:
    def __init__(self, model_path=None):
        self.face_cascade = None
        self.shape_detector = AdvancedFaceShapeDetector()
        self.model_loaded = False
        self.model_path = model_path
        
        # Find the best available model if none specified
        if model_path is None:
            self.model_path = self._find_best_model()
        
        # Load face detection cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face detection cascade loaded successfully")
        except Exception as e:
            print(f"❌ Error loading face cascade: {e}")
            return
        
        # Load face shape model
        if self.model_path and os.path.exists(self.model_path):
            self.model_loaded = self.shape_detector.load_model(self.model_path)
            if self.model_loaded:
                print(f"✅ Face shape model loaded: {self.model_path}")
            else:
                print("❌ Failed to load face shape model")
        else:
            print("❌ No trained model found. Please train a model first.")
    
    def _find_best_model(self):
        """Find the best available trained model"""
        model_patterns = [
            'advanced_face_shape_model_efficient*.h5',
            'advanced_face_shape_model_resnet*.h5',
            'advanced_face_shape_model_custom*.h5',
            'best_efficient_face_shape_model.h5',
            'best_resnet_face_shape_model.h5',
            'best_custom_face_shape_model.h5'
        ]
        
        import glob
        for pattern in model_patterns:
            models = glob.glob(pattern)
            if models:
                # Get the most recent model
                return max(models, key=os.path.getctime)
        
        return None
    
    def detect_faces(self, image):
        """Detect faces in image with optimized parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80),  # Larger minimum size for better quality
            maxSize=(400, 400)  # Maximum size to avoid processing very large faces
        )
        return faces
    
    def extract_face(self, image, face_coords):
        """Extract and preprocess face region"""
        x, y, w, h = face_coords
        
        # Add padding but ensure we don't go out of bounds
        padding = int(0.15 * min(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_img = image[y1:y2, x1:x2]
        return face_img
    
    def get_detailed_recommendations(self, face_shape, confidence, all_predictions):
        """Get comprehensive style recommendations with confidence analysis"""
        
        # Enhanced recommendations based on face shape
        recommendations = {
            'Heart': {
                'description': '💖 Heart-shaped face with wider forehead and narrower chin',
                'hair_styles': [
                    '✨ Side-swept bangs to balance forehead width',
                    '🌊 Long layered cuts that add volume at chin level',
                    '💇 Chin-length bob with soft, textured layers',
                    '🎀 Soft, wispy face-framing pieces',
                    '📐 Asymmetrical cuts that create visual balance'
                ],
                'hair_avoid': [
                    '❌ Center parts that emphasize forehead',
                    '❌ Short, choppy layers above chin',
                    '❌ Severe, straight-across bangs'
                ],
                'beard_styles': [
                    '🧔 Light stubble to define jawline',
                    '🎯 Full goatee to add weight to chin area',
                    '✨ Soul patch for subtle chin emphasis',
                    '📐 Angular beard shapes',
                    '🔷 Extended goatee for length'
                ],
                'beard_avoid': [
                    '❌ Full beards that hide natural jaw shape',
                    '❌ Very wide mustaches that emphasize forehead'
                ]
            },
            'Oval': {
                'description': '🥚 Perfectly balanced oval face - most versatile shape',
                'hair_styles': [
                    '🌟 Almost all hairstyles work beautifully',
                    '💇 Shoulder-length cuts with subtle layers',
                    '✂️ Pixie cuts for bold, chic look',
                    '🌊 Long, flowing waves or straight styles',
                    '🎭 Dramatic updos and ponytails',
                    '📏 Blunt cuts and geometric bobs'
                ],
                'hair_avoid': [
                    '⚠️ Very few restrictions - experiment freely!'
                ],
                'beard_styles': [
                    '🧔 Full beard for distinguished look',
                    '✨ 5 o\'clock shadow for casual elegance',
                    '👨 Classic mustache styles',
                    '🎯 Various goatee styles',
                    '😊 Clean shaven to highlight features',
                    '🔶 Circle beards and anchor styles'
                ],
                'beard_avoid': [
                    '⚠️ Most styles work - choose based on personal preference!'
                ]
            },
            'Round': {
                'description': '🌙 Round face with soft curves and full cheeks',
                'hair_styles': [
                    '📏 Layered cuts with height at crown',
                    '📐 Deep side parts for asymmetry',
                    '✨ Angular styles that add definition',
                    '🔺 Long, straight styles to elongate',
                    '⬆️ High ponytails and top knots',
                    '💇 A-line bobs that angle downward'
                ],
                'hair_avoid': [
                    '❌ Blunt, chin-length cuts',
                    '❌ Center parts with rounded styles',
                    '❌ Very short, curved cuts that emphasize roundness'
                ],
                'beard_styles': [
                    '📐 Angular, structured beard shapes',
                    '🎯 Goatee to elongate face',
                    '✨ Extended goatee with sharp lines',
                    '🔺 Pointed beard styles',
                    '📏 Vertical beard lines'
                ],
                'beard_avoid': [
                    '❌ Round, full beards that add width',
                    '❌ Circular mustaches',
                    '❌ Chops that widen the face'
                ]
            },
            'Square': {
                'description': '⬛ Strong, angular features with prominent jaw',
                'hair_styles': [
                    '🌊 Soft, wavy textures to soften angles',
                    '💫 Side-swept styles with movement',
                    '🌀 Layered cuts that add softness',
                    '📐 Asymmetrical styles',
                    '✨ Face-framing layers',
                    '🎀 Soft, romantic updos'
                ],
                'hair_avoid': [
                    '❌ Blunt cuts that emphasize jaw',
                    '❌ Severe, geometric styles',
                    '❌ Center parts with straight hair',
                    '❌ Very short, angular cuts'
                ],
                'beard_styles': [
                    '🔄 Rounded beard shapes to soften jaw',
                    '⭕ Circle beard for balance',
                    '✨ Light stubble for subtle softening',
                    '🌙 Curved, flowing beard lines',
                    '💫 Soft mustache styles'
                ],
                'beard_avoid': [
                    '❌ Sharp, angular beard shapes',
                    '❌ Square-shaped facial hair',
                    '❌ Harsh, geometric lines'
                ]
            },
            'Oblong': {
                'description': '📏 Elongated face that\'s longer than wide',
                'hair_styles': [
                    '💇 Bangs to shorten face length',
                    '📏 Shoulder-length bobs for width',
                    '🌊 Wide, voluminous styles',
                    '🔄 Curly or wavy textures',
                    '📐 Side parts with volume',
                    '🎀 Styles that add width at temples'
                ],
                'hair_avoid': [
                    '❌ Long, straight styles that elongate',
                    '❌ Center parts that emphasize length',
                    '❌ Very short cuts',
                    '❌ Styles that add height'
                ],
                'beard_styles': [
                    '🧔 Full beard to add width',
                    '📐 Mutton chops for vintage width',
                    '📏 Wide mustache for horizontal balance',
                    '🔄 Horizontal beard lines',
                    '⬜ Box-shaped beards'
                ],
                'beard_avoid': [
                    '❌ Goatees that add length',
                    '❌ Very thin mustaches',
                    '❌ Vertical beard lines'
                ]
            }
        }
        
        base_rec = recommendations.get(face_shape, {
            'description': '🤔 Unique face shape - consult a professional stylist',
            'hair_styles': ['Professional consultation recommended'],
            'hair_avoid': ['Consult with a stylist'],
            'beard_styles': ['Professional consultation recommended'],
            'beard_avoid': ['Consult with a stylist']
        })
        
        # Add confidence-based recommendations
        if confidence < 0.6:
            base_rec['confidence_note'] = f"⚠️ Moderate confidence ({confidence:.1%}). Consider the top 2 predictions."
        elif confidence < 0.8:
            base_rec['confidence_note'] = f"✅ Good confidence ({confidence:.1%}). Recommendations are reliable."
        else:
            base_rec['confidence_note'] = f"🎯 High confidence ({confidence:.1%}). Recommendations are very reliable."
        
        # Add alternative predictions if confidence is low
        if confidence < 0.7:
            sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_predictions) > 1:
                second_shape, second_conf = sorted_predictions[1]
                if second_conf > 0.2:  # If second prediction is significant
                    base_rec['alternative'] = {
                        'shape': second_shape,
                        'confidence': second_conf,
                        'note': f"Also consider styles for {second_shape} face shape ({second_conf:.1%} confidence)"
                    }
        
        return base_rec
    
    def detect_from_camera(self, camera_index=0, save_detections=True):
        """Real-time face shape detection with enhanced features"""
        if not self.model_loaded:
            print("❌ Model not loaded. Cannot proceed.")
            return
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Advanced Face Shape Detection Camera")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Toggle recommendations display")
        print("  'c' - Clear detection history")
        print("=" * 50)
        
        show_recommendations = True
        detection_history = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces every few frames for performance
            if frame_count % 3 == 0:  # Process every 3rd frame
                faces = self.detect_faces(frame)
                
                for face_idx, (x, y, w, h) in enumerate(faces):
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Extract and predict face shape
                    face_img = self.extract_face(frame, (x, y, w, h))
                    
                    if face_img.size > 0:
                        try:
                            # Convert BGR to RGB for prediction
                            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            
                            # Predict face shape
                            face_shape, confidence, all_predictions = self.shape_detector.predict_face_shape(face_rgb)
                            
                            # Store detection
                            detection_info = {
                                'face_shape': face_shape,
                                'confidence': confidence,
                                'timestamp': datetime.now(),
                                'predictions': all_predictions
                            }
                            detection_history.append(detection_info)
                            
                            # Keep only recent detections
                            if len(detection_history) > 10:
                                detection_history.pop(0)
                            
                            # Display main prediction
                            text = f"{face_shape}: {confidence:.1%}"
                            cv2.putText(frame, text, (x, y-15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Display confidence bar
                            bar_width = int(w * confidence)
                            cv2.rectangle(frame, (x, y-10), (x + bar_width, y-5), (0, 255, 0), -1)
                            cv2.rectangle(frame, (x, y-10), (x + w, y-5), (255, 255, 255), 1)
                            
                            if show_recommendations:
                                # Get recommendations
                                recommendations = self.get_detailed_recommendations(
                                    face_shape, confidence, all_predictions
                                )
                                
                                # Display key recommendations
                                y_offset = y + h + 30
                                cv2.putText(frame, "Top Recommendations:", (x, y_offset), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Hair recommendation
                                if recommendations['hair_styles']:
                                    hair_rec = recommendations['hair_styles'][0][:50] + "..."
                                    cv2.putText(frame, f"Hair: {hair_rec}", (x, y_offset + 25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Beard recommendation
                                if recommendations['beard_styles']:
                                    beard_rec = recommendations['beard_styles'][0][:50] + "..."
                                    cv2.putText(frame, f"Beard: {beard_rec}", (x, y_offset + 50), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Confidence note
                                cv2.putText(frame, recommendations.get('confidence_note', ''), 
                                          (x, y_offset + 75), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        except Exception as e:
                            print(f"❌ Prediction error: {e}")
                            cv2.putText(frame, "Prediction Error", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detection_history)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Advanced Face Shape Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if save_detections:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'face_detection_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved as: {filename}")
            elif key == ord('r'):
                show_recommendations = not show_recommendations
                print(f"📋 Recommendations display: {'ON' if show_recommendations else 'OFF'}")
            elif key == ord('c'):
                detection_history.clear()
                print("🗑️ Detection history cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        if detection_history:
            print("\n📊 Detection Summary:")
            shape_counts = {}
            for detection in detection_history:
                shape = detection['face_shape']
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
            for shape, count in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {shape}: {count} detections")
    
    def detect_from_image(self, image_path, save_result=True):
        """Detect face shape from image with detailed analysis"""
        if not self.model_loaded:
            print("❌ Model not loaded. Cannot proceed.")
            return []
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("⚠️ No faces detected in the image")
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
                    face_shape, confidence, all_predictions = self.shape_detector.predict_face_shape(face_rgb)
                    
                    # Get detailed recommendations
                    recommendations = self.get_detailed_recommendations(
                        face_shape, confidence, all_predictions
                    )
                    
                    result = {
                        'face_number': i + 1,
                        'face_shape': face_shape,
                        'confidence': confidence,
                        'all_predictions': all_predictions,
                        'recommendations': recommendations,
                        'coordinates': (x, y, w, h)
                    }
                    results.append(result)
                    
                    # Draw enhanced annotations
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Main prediction
                    text = f"{face_shape}: {confidence:.1%}"
                    cv2.putText(image, text, (x, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Confidence bar
                    bar_width = int(w * confidence)
                    cv2.rectangle(image, (x, y-10), (x + bar_width, y-5), (0, 255, 0), -1)
                    cv2.rectangle(image, (x, y-10), (x + w, y-5), (255, 255, 255), 1)
                    
                    # Top 2 predictions if confidence is low
                    if confidence < 0.8:
                        sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                        if len(sorted_preds) > 1:
                            second_shape, second_conf = sorted_preds[1]
                            cv2.putText(image, f"Alt: {second_shape} ({second_conf:.1%})", 
                                      (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                except Exception as e:
                    print(f"❌ Error processing face {i+1}: {e}")
        
        # Save result image
        if save_result and results:
            result_path = f'result_advanced_{os.path.basename(image_path)}'
            cv2.imwrite(result_path, image)
            print(f"💾 Result saved as: {result_path}")
        
        return results

def main():
    """Main interface for advanced camera detector"""
    print("🎭 Advanced Face Shape Detection")
    print("=" * 50)
    
    detector = AdvancedCameraDetector()
    
    if not detector.model_loaded:
        print("❌ No trained model available.")
        print("Please train a model first using: python advanced_face_detector.py")
        return
    
    while True:
        print("\n🎯 Choose an option:")
        print("1. 📸 Analyze image file")
        print("2. 📹 Real-time camera detection")
        print("3. 📊 Model information")
        print("4. ❌ Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip().strip('"')
            results = detector.detect_from_image(image_path)
            
            if results:
                print(f"\n🎯 Analysis Results for {os.path.basename(image_path)}:")
                print("=" * 60)
                
                for result in results:
                    rec = result['recommendations']
                    print(f"\n👤 Face {result['face_number']}:")
                    print(f"   Shape: {result['face_shape']}")
                    print(f"   Confidence: {result['confidence']:.1%}")
                    print(f"   Description: {rec['description']}")
                    print(f"   {rec.get('confidence_note', '')}")
                    
                    print(f"\n💇 Top Hair Recommendations:")
                    for i, style in enumerate(rec['hair_styles'][:3], 1):
                        print(f"   {i}. {style}")
                    
                    print(f"\n🧔 Top Beard Recommendations:")
                    for i, style in enumerate(rec['beard_styles'][:3], 1):
                        print(f"   {i}. {style}")
                    
                    if 'alternative' in rec:
                        alt = rec['alternative']
                        print(f"\n🔄 Alternative: {alt['note']}")
        
        elif choice == '2':
            camera_idx = input("Enter camera index [0]: ").strip() or "0"
            try:
                detector.detect_from_camera(int(camera_idx))
            except ValueError:
                print("❌ Invalid camera index")
        
        elif choice == '3':
            if detector.model_loaded:
                print(f"\n📊 Model Information:")
                print(f"   Model: {detector.shape_detector.model_name}")
                print(f"   Input size: {detector.shape_detector.img_size}")
                print(f"   Face shapes: {', '.join(detector.shape_detector.face_shapes)}")
                print(f"   Model file: {detector.model_path}")
            else:
                print("❌ No model loaded")
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
