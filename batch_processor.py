"""
Batch processing script for face shape detection
Process multiple images at once and generate a report
"""

import os
import cv2
import numpy as np
from lightweight_detector import LightweightFaceShapeDetector
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

class BatchFaceShapeProcessor:
    def __init__(self, model_path='lightweight_face_shape_model.pkl'):
        self.face_cascade = None
        self.shape_detector = LightweightFaceShapeDetector()
        self.model_loaded = False
        
        # Load face detection cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face detection cascade loaded successfully")
        except Exception as e:
            print(f"❌ Error loading face cascade: {e}")
            return
        
        # Load face shape model
        if os.path.exists(model_path):
            self.model_loaded = self.shape_detector.load_model(model_path)
            if self.model_loaded:
                print("✅ Face shape model loaded successfully")
            else:
                print("❌ Failed to load face shape model")
        else:
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first using lightweight_detector.py")
    
    def detect_faces(self, image):
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        return faces
    
    def extract_face(self, image, face_coords):
        """Extract face region from image"""
        x, y, w, h = face_coords
        padding = int(0.1 * min(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_img = image[y1:y2, x1:x2]
        return face_img
    
    def process_single_image(self, image_path):
        """Process a single image and return results"""
        if not self.model_loaded:
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return {
                'image_path': image_path,
                'faces_detected': 0,
                'results': []
            }
        
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
                    
                    results.append({
                        'face_number': i + 1,
                        'face_shape': face_shape,
                        'confidence': confidence,
                        'coordinates': (x, y, w, h)
                    })
                    
                except Exception as e:
                    print(f"❌ Error processing face {i+1} in {image_path}: {e}")
        
        return {
            'image_path': image_path,
            'faces_detected': len(faces),
            'results': results
        }
    
    def process_directory(self, input_dir, output_dir=None):
        """Process all images in a directory"""
        if not os.path.exists(input_dir):
            print(f"❌ Directory not found: {input_dir}")
            return
        
        if output_dir is None:
            output_dir = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return
        
        print(f"🔍 Found {len(image_files)} images to process")
        
        # Process images
        all_results = []
        processed_count = 0
        
        for i, image_path in enumerate(image_files):
            print(f"📸 Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self.process_single_image(image_path)
            if result:
                all_results.append(result)
                processed_count += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(image_files)} images processed")
        
        # Generate report
        self.generate_report(all_results, output_dir)
        
        print(f"✅ Batch processing completed!")
        print(f"📊 Processed: {processed_count}/{len(image_files)} images")
        print(f"📁 Results saved to: {output_dir}")
    
    def generate_report(self, results, output_dir):
        """Generate CSV and HTML reports"""
        # CSV Report
        csv_path = os.path.join(output_dir, 'face_shape_results.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'filename', 'faces_detected', 'face_number', 
                         'face_shape', 'confidence', 'x', 'y', 'width', 'height']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                image_path = result['image_path']
                filename = os.path.basename(image_path)
                faces_detected = result['faces_detected']
                
                if result['results']:
                    for face_result in result['results']:
                        x, y, w, h = face_result['coordinates']
                        writer.writerow({
                            'image_path': image_path,
                            'filename': filename,
                            'faces_detected': faces_detected,
                            'face_number': face_result['face_number'],
                            'face_shape': face_result['face_shape'],
                            'confidence': face_result['confidence'],
                            'x': x, 'y': y, 'width': w, 'height': h
                        })
                else:
                    # No faces detected
                    writer.writerow({
                        'image_path': image_path,
                        'filename': filename,
                        'faces_detected': 0,
                        'face_number': 0,
                        'face_shape': 'None',
                        'confidence': 0,
                        'x': 0, 'y': 0, 'width': 0, 'height': 0
                    })
        
        # Generate statistics
        self.generate_statistics(results, output_dir)
        
        # Generate HTML report
        self.generate_html_report(results, output_dir)
    
    def generate_statistics(self, results, output_dir):
        """Generate statistics and charts"""
        # Collect face shape counts
        face_shape_counts = {}
        total_faces = 0
        images_with_faces = 0
        
        for result in results:
            if result['faces_detected'] > 0:
                images_with_faces += 1
                for face_result in result['results']:
                    face_shape = face_result['face_shape']
                    face_shape_counts[face_shape] = face_shape_counts.get(face_shape, 0) + 1
                    total_faces += 1
        
        # Create bar chart
        if face_shape_counts:
            plt.figure(figsize=(10, 6))
            shapes = list(face_shape_counts.keys())
            counts = list(face_shape_counts.values())
            
            bars = plt.bar(shapes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFDAB9'])
            plt.title('Face Shape Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Face Shape', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'face_shape_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save statistics to text file
        stats_path = os.path.join(output_dir, 'statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("Face Shape Detection - Batch Processing Statistics\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Total Images Processed: {len(results)}\n")
            f.write(f"Images with Faces Detected: {images_with_faces}\n")
            f.write(f"Total Faces Detected: {total_faces}\n\n")
            
            if face_shape_counts:
                f.write("Face Shape Distribution:\n")
                f.write("-" * 25 + "\n")
                for shape, count in sorted(face_shape_counts.items()):
                    percentage = (count / total_faces) * 100
                    f.write(f"{shape:10}: {count:3} ({percentage:5.1f}%)\n")
    
    def generate_html_report(self, results, output_dir):
        """Generate HTML report"""
        html_path = os.path.join(output_dir, 'report.html')
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Shape Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #2E86AB; color: white; padding: 20px; border-radius: 10px; }
                .summary { background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }
                .result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .face-shape { font-weight: bold; color: #2E86AB; }
                .confidence { color: #666; }
                .no-face { color: #999; font-style: italic; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 Face Shape Detection Report</h1>
                <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        """
        
        # Add summary
        total_images = len(results)
        images_with_faces = sum(1 for r in results if r['faces_detected'] > 0)
        total_faces = sum(len(r['results']) for r in results)
        
        html_content += f"""
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Total Images:</strong> {total_images}</p>
                <p><strong>Images with Faces:</strong> {images_with_faces}</p>
                <p><strong>Total Faces Detected:</strong> {total_faces}</p>
            </div>
            
            <h2>📋 Detailed Results</h2>
        """
        
        # Add individual results
        for result in results:
            filename = os.path.basename(result['image_path'])
            faces_count = result['faces_detected']
            
            html_content += f"""
            <div class="result">
                <h3>📸 {filename}</h3>
                <p><strong>Faces Detected:</strong> {faces_count}</p>
            """
            
            if result['results']:
                for face_result in result['results']:
                    html_content += f"""
                    <p>👤 <span class="face-shape">Face {face_result['face_number']}: {face_result['face_shape']}</span> 
                    <span class="confidence">(Confidence: {face_result['confidence']:.1%})</span></p>
                    """
            else:
                html_content += '<p class="no-face">No faces detected</p>'
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Batch Face Shape Detection')
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--model', '-m', default='lightweight_face_shape_model.pkl',
                       help='Path to the trained model file')
    
    args = parser.parse_args()
    
    print("🎭 Batch Face Shape Detection")
    print("=" * 40)
    
    # Initialize processor
    processor = BatchFaceShapeProcessor(args.model)
    
    if not processor.model_loaded:
        print("❌ Cannot proceed without a trained model")
        return
    
    # Process directory
    processor.process_directory(args.input_dir, args.output)

if __name__ == "__main__":
    # If no command line arguments, use interactive mode
    import sys
    if len(sys.argv) == 1:
        print("🎭 Batch Face Shape Detection")
        print("=" * 40)
        
        input_dir = input("Enter directory path containing images: ").strip().strip('"')
        if not input_dir:
            print("❌ No directory specified")
            exit(1)
        
        processor = BatchFaceShapeProcessor()
        if processor.model_loaded:
            processor.process_directory(input_dir)
        else:
            print("❌ Cannot proceed without a trained model")
    else:
        main()
