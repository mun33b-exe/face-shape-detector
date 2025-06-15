"""
Hairstyle and Beard Recommendation Engine

This module provides style recommendations based on face shape analysis.
"""

import json
import os
from typing import Dict, List, Tuple
import random

class StyleRecommendationEngine:
    """Recommend hairstyles and beard styles based on face shape"""
    
    def __init__(self):
        self.recommendations = self._load_style_database()
    
    def _load_style_database(self) -> Dict:
        """Load style recommendations database"""
        # This would typically load from a JSON file or database
        # For now, we'll define recommendations inline
        
        return {
            'oval': {
                'hairstyles': [
                    {
                        'name': 'Classic Side Part',
                        'description': 'Timeless and professional look',
                        'difficulty': 'easy',
                        'filter_file': 'side_part_oval.png'
                    },
                    {
                        'name': 'Textured Quiff',
                        'description': 'Modern and stylish with volume',
                        'difficulty': 'medium',
                        'filter_file': 'quiff_oval.png'
                    },
                    {
                        'name': 'Slicked Back',
                        'description': 'Sophisticated and sleek',
                        'difficulty': 'easy',
                        'filter_file': 'slicked_back_oval.png'
                    }
                ],
                'beards': [
                    {
                        'name': 'Full Beard',
                        'description': 'Classic full coverage',
                        'maintenance': 'high',
                        'filter_file': 'full_beard_oval.png'
                    },
                    {
                        'name': 'Goatee',
                        'description': 'Focused on chin area',
                        'maintenance': 'medium',
                        'filter_file': 'goatee_oval.png'
                    },
                    {
                        'name': 'Stubble',
                        'description': 'Light, casual look',
                        'maintenance': 'low',
                        'filter_file': 'stubble_oval.png'
                    }
                ]
            },
            'round': {
                'hairstyles': [
                    {
                        'name': 'High Fade',
                        'description': 'Creates vertical lines, adds height',
                        'difficulty': 'medium',
                        'filter_file': 'high_fade_round.png'
                    },
                    {
                        'name': 'Pompadour',
                        'description': 'Adds height and elongates face',
                        'difficulty': 'hard',
                        'filter_file': 'pompadour_round.png'
                    },
                    {
                        'name': 'Angular Fringe',
                        'description': 'Sharp lines to contrast curves',
                        'difficulty': 'medium',
                        'filter_file': 'angular_fringe_round.png'
                    }
                ],
                'beards': [
                    {
                        'name': 'Chin Strap',
                        'description': 'Defines jawline',
                        'maintenance': 'medium',
                        'filter_file': 'chin_strap_round.png'
                    },
                    {
                        'name': 'Soul Patch',
                        'description': 'Minimal, adds definition',
                        'maintenance': 'low',
                        'filter_file': 'soul_patch_round.png'
                    }
                ]
            },
            'square': {
                'hairstyles': [
                    {
                        'name': 'Textured Crop',
                        'description': 'Softens angular features',
                        'difficulty': 'easy',
                        'filter_file': 'textured_crop_square.png'
                    },
                    {
                        'name': 'Messy Waves',
                        'description': 'Adds softness and movement',
                        'difficulty': 'medium',
                        'filter_file': 'messy_waves_square.png'
                    },
                    {
                        'name': 'Long Layers',
                        'description': 'Creates flowing lines',
                        'difficulty': 'easy',
                        'filter_file': 'long_layers_square.png'
                    }
                ],
                'beards': [
                    {
                        'name': 'Rounded Beard',
                        'description': 'Softens square jawline',
                        'maintenance': 'high',
                        'filter_file': 'rounded_beard_square.png'
                    },
                    {
                        'name': 'Circle Beard',
                        'description': 'Rounded mustache and goatee',
                        'maintenance': 'medium',
                        'filter_file': 'circle_beard_square.png'
                    }
                ]
            },
            'heart': {
                'hairstyles': [
                    {
                        'name': 'Side Swept Bangs',
                        'description': 'Balances wide forehead',
                        'difficulty': 'easy',
                        'filter_file': 'side_swept_heart.png'
                    },
                    {
                        'name': 'Chin-Length Bob',
                        'description': 'Adds width at jawline',
                        'difficulty': 'medium',
                        'filter_file': 'chin_bob_heart.png'
                    },
                    {
                        'name': 'Low Fade',
                        'description': 'Keeps width at temples',
                        'difficulty': 'medium',
                        'filter_file': 'low_fade_heart.png'
                    }
                ],
                'beards': [
                    {
                        'name': 'Full Chin Beard',
                        'description': 'Adds weight to lower face',
                        'maintenance': 'high',
                        'filter_file': 'full_chin_heart.png'
                    },
                    {
                        'name': 'Extended Goatee',
                        'description': 'Widens chin area',
                        'maintenance': 'medium',
                        'filter_file': 'extended_goatee_heart.png'
                    }
                ]
            },
            'oblong': {
                'hairstyles': [
                    {
                        'name': 'Fringe/Bangs',
                        'description': 'Shortens face length',
                        'difficulty': 'easy',
                        'filter_file': 'fringe_oblong.png'
                    },
                    {
                        'name': 'Layered Cut',
                        'description': 'Adds width and volume',
                        'difficulty': 'medium',
                        'filter_file': 'layered_oblong.png'
                    },
                    {
                        'name': 'Wide Caesar',
                        'description': 'Creates horizontal emphasis',
                        'difficulty': 'easy',
                        'filter_file': 'wide_caesar_oblong.png'
                    }
                ],
                'beards': [
                    {
                        'name': 'Horizontal Mustache',
                        'description': 'Adds width to mid-face',
                        'maintenance': 'low',
                        'filter_file': 'horizontal_mustache_oblong.png'
                    },
                    {
                        'name': 'Wide Beard',
                        'description': 'Balances face length',
                        'maintenance': 'high',
                        'filter_file': 'wide_beard_oblong.png'
                    }
                ]
            }
        }
    
    def get_recommendations(self, face_shape: str, style_type: str = 'both') -> Dict:
        """
        Get style recommendations for a specific face shape
        
        Args:
            face_shape: Detected face shape (oval, round, square, heart, oblong)
            style_type: 'hairstyles', 'beards', or 'both'
        
        Returns:
            Dictionary containing recommended styles
        """
        if face_shape.lower() not in self.recommendations:
            return {'error': f'No recommendations available for face shape: {face_shape}'}
        
        shape_data = self.recommendations[face_shape.lower()]
        
        if style_type == 'hairstyles':
            return {'hairstyles': shape_data.get('hairstyles', [])}
        elif style_type == 'beards':
            return {'beards': shape_data.get('beards', [])}
        else:
            return {
                'hairstyles': shape_data.get('hairstyles', []),
                'beards': shape_data.get('beards', [])
            }
    
    def get_top_recommendations(self, face_shape: str, count: int = 3) -> Dict:
        """Get top N recommendations for each style type"""
        recommendations = self.get_recommendations(face_shape)
        
        if 'error' in recommendations:
            return recommendations
        
        return {
            'top_hairstyles': recommendations.get('hairstyles', [])[:count],
            'top_beards': recommendations.get('beards', [])[:count]
        }
    
    def get_random_recommendation(self, face_shape: str, style_type: str) -> Dict:
        """Get a random recommendation for variety"""
        recommendations = self.get_recommendations(face_shape, style_type)
        
        if 'error' in recommendations:
            return recommendations
        
        if style_type == 'hairstyles' and recommendations.get('hairstyles'):
            return {'recommendation': random.choice(recommendations['hairstyles'])}
        elif style_type == 'beards' and recommendations.get('beards'):
            return {'recommendation': random.choice(recommendations['beards'])}
        else:
            return {'error': 'No recommendations available'}
    
    def save_user_preference(self, user_id: str, face_shape: str, 
                           style_name: str, rating: int):
        """Save user preferences for machine learning improvements"""
        # This would typically save to a database
        # For now, we'll just log it
        preference_data = {
            'user_id': user_id,
            'face_shape': face_shape,
            'style_name': style_name,
            'rating': rating
        }
        
        # Log preference (in real app, save to database)
        print(f"User preference saved: {preference_data}")
    
    def get_style_details(self, style_name: str, face_shape: str) -> Dict:
        """Get detailed information about a specific style"""
        shape_data = self.recommendations.get(face_shape.lower(), {})
        
        # Search in hairstyles
        for style in shape_data.get('hairstyles', []):
            if style['name'].lower() == style_name.lower():
                return {
                    'type': 'hairstyle',
                    'details': style,
                    'face_shape': face_shape
                }
        
        # Search in beards
        for style in shape_data.get('beards', []):
            if style['name'].lower() == style_name.lower():
                return {
                    'type': 'beard',
                    'details': style,
                    'face_shape': face_shape
                }
        
        return {'error': 'Style not found'}

def main():
    """Test the recommendation engine"""
    engine = StyleRecommendationEngine()
    
    # Test recommendations for different face shapes
    test_shapes = ['oval', 'round', 'square', 'heart', 'oblong']
    
    for shape in test_shapes:
        print(f"\n--- Recommendations for {shape.upper()} face shape ---")
        recommendations = engine.get_top_recommendations(shape, 2)
        
        print("Top Hairstyles:")
        for style in recommendations.get('top_hairstyles', []):
            print(f"  - {style['name']}: {style['description']}")
        
        print("Top Beard Styles:")
        for style in recommendations.get('top_beards', []):
            print(f"  - {style['name']}: {style['description']}")

if __name__ == "__main__":
    main()
