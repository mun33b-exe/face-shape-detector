"""
Simple training script for face shape detection model
Run this first to train your model on the dataset
"""

from face_shape_detector import FaceShapeDetector
import os

def main():
    print("🎭 Face Shape Detection Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "FaceShape Dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please make sure the dataset folder is in the same directory as this script.")
        return
    
    try:
        # Initialize detector
        print("🔧 Initializing detector...")
        detector = FaceShapeDetector(dataset_path)
        
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        X, y = detector.load_and_preprocess_data()
        
        # Create model
        print("🏗️ Creating model architecture...")
        model = detector.create_model()
        print(f"Model created with {model.count_params():,} parameters")
        
        # Train model
        print("🚀 Starting training...")
        print("This may take 10-30 minutes depending on your hardware...")
        history = detector.train_model(X, y, epochs=30)
        
        # Evaluate model
        print("📈 Evaluating model on test set...")
        accuracy = detector.evaluate_model()
        
        # Save model
        print("💾 Saving model...")
        detector.save_model()
        
        print("=" * 50)
        print(f"✅ Training completed successfully!")
        print(f"📊 Final test accuracy: {accuracy:.2%}")
        print(f"📁 Model saved as: face_shape_model.h5")
        print("🎉 You can now use the model for predictions!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    main()
