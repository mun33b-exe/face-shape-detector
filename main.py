"""
Main Application - AI Face Shape Detection & Styling System

This is the main entry point that combines face detection, shape classification,
style recommendations, and virtual try-on functionality.
"""

import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Optional
import argparse
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.face_detection import FaceDetector, FaceShapeClassifier
    from src.style_recommendations import StyleRecommendationEngine
    from src.virtual_tryon import VirtualTryOnFilter
except ImportError:
    # Fallback for direct execution
    from face_detection import FaceDetector, FaceShapeClassifier
    from style_recommendations import StyleRecommendationEngine
    from virtual_tryon import VirtualTryOnFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceShapeStyleApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.shape_classifier = FaceShapeClassifier()
        self.recommendation_engine = StyleRecommendationEngine()
        self.virtual_tryon = VirtualTryOnFilter()
        
        self.current_image = None
        self.current_analysis = None
        self.current_recommendations = None
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze an image for face shape and get recommendations"""
        
        logger.info("Starting face shape analysis...")
        
        # Analyze face shape
        analysis_result = self.shape_classifier.analyze_face_shape(image, self.face_detector)
        
        if 'error' in analysis_result:
            return analysis_result
        
        face_shape = analysis_result['face_shape']
        logger.info(f"Detected face shape: {face_shape}")
        
        # Get style recommendations
        recommendations = self.recommendation_engine.get_top_recommendations(face_shape, count=3)
        
        # Store current state
        self.current_image = image.copy()
        self.current_analysis = analysis_result
        self.current_recommendations = recommendations
        
        return {
            'face_shape': face_shape,
            'confidence': analysis_result.get('confidence', 0.0),
            'measurements': analysis_result.get('measurements', {}),
            'recommendations': recommendations
        }
    
    def apply_style_filter(self, style_name: str, style_type: str, opacity: float = 0.8) -> Optional[np.ndarray]:
        """Apply a specific style filter to the current image"""
        
        if self.current_image is None or self.current_analysis is None:
            logger.error("No image analyzed yet. Please analyze an image first.")
            return None
        
        # Find the filter filename for the style
        filter_filename = self._get_filter_filename(style_name, style_type)
        if not filter_filename:
            logger.error(f"Filter not found for style: {style_name}")
            return None
        
        # Apply the filter
        face_landmarks = np.array(self.current_analysis['landmarks'])
        face_rect = self.current_analysis['face_rect']
        
        filtered_image = self.virtual_tryon.apply_filter(
            self.current_image, filter_filename, style_type,
            face_landmarks, face_rect, opacity
        )
        
        return filtered_image
    
    def _get_filter_filename(self, style_name: str, style_type: str) -> Optional[str]:
        """Get the filter filename for a specific style"""
        
        if not self.current_recommendations:
            return None
        
        styles_key = 'hairstyles' if style_type == 'hairstyle' else 'beards'
        styles = self.current_recommendations.get(styles_key, [])
        
        for style in styles:
            if style['name'].lower() == style_name.lower():
                return style.get('filter_file')
        
        return None
    
    def run_camera_mode(self):
        """Run the application in real-time camera mode"""
        
        logger.info("Starting camera mode...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        print("\n=== Camera Mode Controls ===")
        print("Press 'q' to quit")
        print("Press 's' to analyze current frame")
        print("Press '1-3' to apply hairstyle filters")
        print("Press '4-6' to apply beard filters")
        print("Press 'r' to reset to original")
        print("Press 'c' to capture and save image")
        
        current_display = None
        analysis_done = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display current frame or filtered result
            display_frame = current_display if current_display is not None else frame
            
            # Add status text
            if analysis_done and self.current_analysis:
                face_shape = self.current_analysis['face_shape']
                cv2.putText(display_frame, f"Face Shape: {face_shape.upper()}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('AI Face Shape & Style Detector', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Analyze current frame
                logger.info("Analyzing current frame...")
                result = self.analyze_image(frame)
                if 'error' not in result:
                    analysis_done = True
                    current_display = frame.copy()
                    self._print_analysis_results(result)
                else:
                    print(f"Analysis failed: {result['error']}")
            
            elif key == ord('r'):
                # Reset to original
                current_display = self.current_image.copy() if self.current_image is not None else frame
            
            elif key == ord('c') and current_display is not None:
                # Capture and save
                timestamp = cv2.getTickCount()
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, current_display)
                print(f"Image saved as: {filename}")
            
            # Apply filters based on key press
            elif analysis_done and self.current_recommendations:
                filtered_result = None
                
                if key in [ord('1'), ord('2'), ord('3')]:
                    # Apply hairstyle filters
                    hairstyles = self.current_recommendations.get('top_hairstyles', [])
                    idx = key - ord('1')
                    if idx < len(hairstyles):
                        style_name = hairstyles[idx]['name']
                        print(f"Applying hairstyle: {style_name}")
                        filtered_result = self.apply_style_filter(style_name, 'hairstyle')
                
                elif key in [ord('4'), ord('5'), ord('6')]:
                    # Apply beard filters
                    beards = self.current_recommendations.get('top_beards', [])
                    idx = key - ord('4')
                    if idx < len(beards):
                        style_name = beards[idx]['name']
                        print(f"Applying beard style: {style_name}")
                        filtered_result = self.apply_style_filter(style_name, 'beard')
                
                if filtered_result is not None:
                    current_display = filtered_result
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_image_mode(self, image_path: str):
        """Run the application on a single image file"""
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        logger.info(f"Processing image: {image_path}")
        
        # Analyze image
        result = self.analyze_image(image)
        if 'error' in result:
            print(f"Analysis failed: {result['error']}")
            return
        
        # Print results
        self._print_analysis_results(result)
        
        # Show original image
        cv2.imshow('Original Image', image)
        
        # Interactive filter application
        self._interactive_filter_mode(image)
    
    def _print_analysis_results(self, result: Dict):
        """Print analysis results in a formatted way"""
        
        print(f"\n{'='*50}")
        print(f"FACE SHAPE ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Detected Face Shape: {result['face_shape'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        recommendations = result['recommendations']
        
        print(f"\n--- TOP HAIRSTYLE RECOMMENDATIONS ---")
        for i, style in enumerate(recommendations.get('top_hairstyles', []), 1):
            print(f"{i}. {style['name']}")
            print(f"   Description: {style['description']}")
            print(f"   Difficulty: {style['difficulty']}")
        
        print(f"\n--- TOP BEARD RECOMMENDATIONS ---")
        for i, style in enumerate(recommendations.get('top_beards', []), 1):
            print(f"{i}. {style['name']}")
            print(f"   Description: {style['description']}")
            print(f"   Maintenance: {style['maintenance']}")
    
    def _interactive_filter_mode(self, image: np.ndarray):
        """Interactive mode for applying filters to an image"""
        
        print(f"\n{'='*50}")
        print("INTERACTIVE FILTER MODE")
        print("Press number keys to apply filters, 'q' to quit")
        print(f"{'='*50}")
        
        current_display = image.copy()
        
        while True:
            cv2.imshow('Face Shape & Style Filter', current_display)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                current_display = image.copy()
                print("Reset to original image")
            
            # Apply filters based on recommendations
            elif self.current_recommendations:
                filtered_result = None
                
                if key in [ord('1'), ord('2'), ord('3')]:
                    hairstyles = self.current_recommendations.get('top_hairstyles', [])
                    idx = key - ord('1')
                    if idx < len(hairstyles):
                        style_name = hairstyles[idx]['name']
                        print(f"Applying hairstyle: {style_name}")
                        filtered_result = self.apply_style_filter(style_name, 'hairstyle')
                
                elif key in [ord('4'), ord('5'), ord('6')]:
                    beards = self.current_recommendations.get('top_beards', [])
                    idx = key - ord('4')
                    if idx < len(beards):
                        style_name = beards[idx]['name']
                        print(f"Applying beard style: {style_name}")
                        filtered_result = self.apply_style_filter(style_name, 'beard')
                
                if filtered_result is not None:
                    current_display = filtered_result
        
        cv2.destroyAllWindows()

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='AI Face Shape Detection & Styling System')
    parser.add_argument('--mode', choices=['camera', 'image'], default='camera',
                       help='Run mode: camera for real-time, image for single image')
    parser.add_argument('--image', type=str, help='Path to image file (required for image mode)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize application
    app = FaceShapeStyleApp()
    
    print("AI Face Shape Detection & Styling Recommendation System")
    print("="*60)
    
    try:
        if args.mode == 'camera':
            app.run_camera_mode()
        elif args.mode == 'image':
            if not args.image:
                print("Error: --image argument required for image mode")
                sys.exit(1)
            app.run_image_mode(args.image)
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.debug:
            raise

if __name__ == "__main__":
    main()
