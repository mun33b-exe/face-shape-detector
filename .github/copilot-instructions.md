<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Face Shape Detection & Styling Recommendation System

This project implements an AI-powered system for face shape detection and hairstyle/beard recommendations with virtual try-on capabilities.

## Project Structure
- `src/` - Main source code
- `models/` - Trained ML models and weights
- `datasets/` - Training and test data
- `filters/` - Hairstyle and beard overlay images
- `utils/` - Helper functions and utilities

## Key Technologies
- Computer Vision: OpenCV, dlib, MediaPipe
- Machine Learning: TensorFlow/Keras for face shape classification
- Image Processing: PIL, scikit-image for filter overlays
- Face Detection: face-recognition library with dlib

## Development Guidelines
- Follow modular architecture with separate components for detection, classification, and styling
- Use pre-trained models where possible to reduce training time
- Implement proper error handling for camera access and image processing
- Create reusable functions for different face shapes and styling options
- Add comprehensive logging for debugging ML model performance
