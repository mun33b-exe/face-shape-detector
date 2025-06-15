"""
Virtual Try-On Filter System

This module handles applying hairstyle and beard filters to user photos
using image overlay techniques and facial landmark alignment.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, List
import logging

# Optional imports
try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Some image processing features may be limited.")

logger = logging.getLogger(__name__)

class VirtualTryOnFilter:
    """Apply hairstyle and beard filters to face images"""
    
    def __init__(self, filters_path: str = "filters"):
        self.filters_path = filters_path
        self.hairstyles_path = os.path.join(filters_path, "hairstyles")
        self.beards_path = os.path.join(filters_path, "beards")
        
        # Ensure filter directories exist
        os.makedirs(self.hairstyles_path, exist_ok=True)
        os.makedirs(self.beards_path, exist_ok=True)
    
    def load_filter_image(self, filter_filename: str, filter_type: str) -> Optional[np.ndarray]:
        """Load a filter image from the appropriate directory"""
        if filter_type == 'hairstyle':
            filter_path = os.path.join(self.hairstyles_path, filter_filename)
        elif filter_type == 'beard':
            filter_path = os.path.join(self.beards_path, filter_filename)
        else:
            logger.error(f"Unknown filter type: {filter_type}")
            return None
        
        if not os.path.exists(filter_path):
            logger.warning(f"Filter image not found: {filter_path}")
            return None
        
        # Load image with alpha channel for transparency
        filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
        return filter_img
    
    def align_filter_to_face(self, filter_img: np.ndarray, 
                           face_landmarks: np.ndarray,
                           face_rect: Tuple[int, int, int, int],
                           filter_type: str) -> Optional[np.ndarray]:
        """Align and scale filter to match face proportions"""
        
        if filter_img is None or face_landmarks is None:
            return None
        
        # Get face dimensions
        face_left, face_top, face_right, face_bottom = face_rect
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        
        # Define anchor points based on filter type
        if filter_type == 'hairstyle':
            # Use forehead and temple points for hairstyle alignment
            # Landmarks: 17-26 are eyebrow/forehead area
            anchor_points = self._get_hairstyle_anchors(face_landmarks)
            target_width = int(face_width * 1.2)  # Slightly wider than face
            target_height = int(face_height * 0.8)  # Hair covers top portion
            
        elif filter_type == 'beard':
            # Use jaw and chin points for beard alignment
            # Landmarks: 0-16 are jawline, 6-10 are lower jaw/chin
            anchor_points = self._get_beard_anchors(face_landmarks)
            target_width = int(face_width * 0.9)  # Slightly narrower than face width
            target_height = int(face_height * 0.6)  # Beard covers lower portion
        
        else:
            return None
        
        # Resize filter image
        resized_filter = cv2.resize(filter_img, (target_width, target_height))
        
        # Calculate position
        if filter_type == 'hairstyle':
            # Position hair above eyebrows
            x_offset = face_left - (target_width - face_width) // 2
            y_offset = face_top - int(target_height * 0.7)
        else:  # beard
            # Position beard on lower face
            x_offset = face_left + (face_width - target_width) // 2
            y_offset = face_top + int(face_height * 0.4)
        
        return resized_filter, (x_offset, y_offset)
    
    def _get_hairstyle_anchors(self, landmarks: np.ndarray) -> List[Tuple[int, int]]:
        """Get key points for hairstyle alignment"""
        # Forehead/temple area landmarks
        return [
            tuple(landmarks[17]),  # Left eyebrow start
            tuple(landmarks[26]),  # Right eyebrow start
            tuple(landmarks[19]),  # Left eyebrow peak
            tuple(landmarks[24]),  # Right eyebrow peak
        ]
    
    def _get_beard_anchors(self, landmarks: np.ndarray) -> List[Tuple[int, int]]:
        """Get key points for beard alignment"""
        # Jawline landmarks
        return [
            tuple(landmarks[3]),   # Left jaw
            tuple(landmarks[13]),  # Right jaw
            tuple(landmarks[8]),   # Chin center
            tuple(landmarks[6]),   # Lower left jaw
            tuple(landmarks[10]),  # Lower right jaw
        ]
    
    def blend_filter_with_image(self, base_image: np.ndarray, 
                               filter_data: Tuple[np.ndarray, Tuple[int, int]],
                               blend_mode: str = 'normal',
                               opacity: float = 0.8) -> np.ndarray:
        """Blend filter with base image using various blending modes"""
        
        filter_img, (x_offset, y_offset) = filter_data
        result_image = base_image.copy()
        
        # Handle negative offsets
        if x_offset < 0:
            filter_img = filter_img[:, abs(x_offset):]
            x_offset = 0
        if y_offset < 0:
            filter_img = filter_img[abs(y_offset):, :]
            y_offset = 0
        
        # Get dimensions
        base_h, base_w = base_image.shape[:2]
        filter_h, filter_w = filter_img.shape[:2]
        
        # Adjust filter size if it exceeds base image bounds
        end_x = min(x_offset + filter_w, base_w)
        end_y = min(y_offset + filter_h, base_h)
        
        filter_w = end_x - x_offset
        filter_h = end_y - y_offset
        
        if filter_w <= 0 or filter_h <= 0:
            return result_image
        
        # Crop filter to fit
        filter_img = filter_img[:filter_h, :filter_w]
        
        # Handle alpha channel for transparency
        if filter_img.shape[2] == 4:  # RGBA
            # Extract RGB and alpha channels
            filter_rgb = filter_img[:, :, :3]
            alpha = filter_img[:, :, 3] / 255.0 * opacity
            
            # Apply alpha blending
            for c in range(3):
                result_image[y_offset:end_y, x_offset:end_x, c] = (
                    alpha * filter_rgb[:, :, c] + 
                    (1 - alpha) * result_image[y_offset:end_y, x_offset:end_x, c]
                )
        else:  # RGB
            # Simple overlay with opacity
            roi = result_image[y_offset:end_y, x_offset:end_x]
            blended = cv2.addWeighted(roi, 1-opacity, filter_img, opacity, 0)
            result_image[y_offset:end_y, x_offset:end_x] = blended
        
        return result_image
    
    def apply_filter(self, base_image: np.ndarray, 
                    filter_filename: str,
                    filter_type: str,
                    face_landmarks: np.ndarray,
                    face_rect: Tuple[int, int, int, int],
                    opacity: float = 0.8) -> Optional[np.ndarray]:
        """Complete pipeline to apply a filter to an image"""
        
        # Load filter image
        filter_img = self.load_filter_image(filter_filename, filter_type)
        if filter_img is None:
            return None
        
        # Align filter to face
        aligned_filter = self.align_filter_to_face(
            filter_img, face_landmarks, face_rect, filter_type
        )
        if aligned_filter is None:
            return None
        
        # Blend filter with base image
        result = self.blend_filter_with_image(
            base_image, aligned_filter, opacity=opacity
        )
        
        return result
    
    def apply_multiple_filters(self, base_image: np.ndarray,
                             filters: List[Dict],
                             face_landmarks: np.ndarray,
                             face_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Apply multiple filters (e.g., hairstyle + beard) to one image"""
        
        result_image = base_image.copy()
        
        for filter_info in filters:
            filter_filename = filter_info.get('filename')
            filter_type = filter_info.get('type')
            opacity = filter_info.get('opacity', 0.8)
            
            result_image = self.apply_filter(
                result_image, filter_filename, filter_type,
                face_landmarks, face_rect, opacity
            )
            
            if result_image is None:
                logger.error(f"Failed to apply filter: {filter_filename}")
                return None
        
        return result_image
    
    def create_comparison_view(self, original: np.ndarray, 
                             filtered: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison of original and filtered images"""
        
        # Ensure both images have the same height
        h1, w1 = original.shape[:2]
        h2, w2 = filtered.shape[:2]
        
        target_height = min(h1, h2)
        
        # Resize if necessary
        if h1 != target_height:
            original = cv2.resize(original, (int(w1 * target_height / h1), target_height))
        if h2 != target_height:
            filtered = cv2.resize(filtered, (int(w2 * target_height / h2), target_height))
        
        # Create side-by-side image
        comparison = np.hstack([original, filtered])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, target_height - 20), 
                   font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, 'With Filter', (original.shape[1] + 10, target_height - 20), 
                   font, 0.7, (255, 255, 255), 2)
        
        return comparison

def create_sample_filters():
    """Create sample filter placeholders (colored rectangles for testing)"""
    
    filter_types = ['hairstyles', 'beards']
    colors = {
        'hairstyles': [(139, 69, 19), (101, 67, 33), (62, 39, 35)],  # Brown tones
        'beards': [(101, 67, 33), (139, 69, 19), (160, 82, 45)]       # Brown/tan tones
    }
    
    for filter_type in filter_types:
        folder_path = f"filters/{filter_type}"
        os.makedirs(folder_path, exist_ok=True)
        
        for i, color in enumerate(colors[filter_type]):
            # Create a sample filter image (colored rectangle with transparency)
            if filter_type == 'hairstyles':
                img = np.zeros((200, 300, 4), dtype=np.uint8)  # RGBA
                img[:, :, :3] = color  # RGB channels
                img[:, :, 3] = 180     # Alpha channel (transparency)
                filename = f"sample_hair_{i+1}.png"
            else:
                img = np.zeros((150, 200, 4), dtype=np.uint8)  # RGBA
                img[:, :, :3] = color  # RGB channels
                img[:, :, 3] = 200     # Alpha channel
                filename = f"sample_beard_{i+1}.png"
            
            cv2.imwrite(os.path.join(folder_path, filename), img)
    
    print("Sample filter images created!")

def main():
    """Test the virtual try-on system"""
    
    # Create sample filters for testing
    create_sample_filters()
    
    # Initialize the filter system
    filter_system = VirtualTryOnFilter()
    
    print("Virtual try-on filter system initialized!")
    print("Sample filters created in 'filters/' directory")
    print("You can replace these with actual hairstyle and beard images")

if __name__ == "__main__":
    main()
