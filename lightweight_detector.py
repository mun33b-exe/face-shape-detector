"""
Lightweight Face Shape Detector using traditional ML approaches
This version works without TensorFlow and provides a good starting point
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt

class LightweightFaceShapeDetector:
    def __init__(self, dataset_path="FaceShape Dataset"):
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.img_size = (64, 64)  # Smaller size for traditional ML
        self.face_shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        
    def extract_features(self, image):
        """Extract features from face image"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        resized = cv2.resize(gray, self.img_size)
        
        # Feature extraction methods
        features = []
        
        # 1. Pixel intensities (flattened)
        pixel_features = resized.flatten() / 255.0
        features.extend(pixel_features)
        
        # 2. Face measurements/ratios
        height, width = resized.shape
        
        # Width to height ratio
        ratio = width / height
        features.append(ratio)
        
        # Upper face width (forehead area)
        upper_third = resized[:height//3, :]
        upper_width = np.sum(upper_third > np.mean(upper_third))
        features.append(upper_width / (width * height // 3))
        
        # Middle face width (cheek area)
        middle_third = resized[height//3:2*height//3, :]
        middle_width = np.sum(middle_third > np.mean(middle_third))
        features.append(middle_width / (width * height // 3))
        
        # Lower face width (jaw area)
        lower_third = resized[2*height//3:, :]
        lower_width = np.sum(lower_third > np.mean(lower_third))
        features.append(lower_width / (width * height // 3))
        
        # Jaw line sharpness (edge detection)
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        features.append(edge_density)
        
        return np.array(features)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the face shape dataset"""
        print("Loading and preprocessing data...")
        
        features_list = []
        labels = []
        
        # Load training data
        train_dir = os.path.join(self.dataset_path, 'training_set')
        for shape in self.face_shapes:
            shape_dir = os.path.join(train_dir, shape)
            if os.path.exists(shape_dir):
                print(f"Processing {shape} images...")
                count = 0
                for img_file in os.listdir(shape_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(shape_dir, img_file)
                        try:
                            # Load and preprocess image
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                # Extract features
                                features = self.extract_features(img)
                                features_list.append(features)
                                labels.append(shape)
                                count += 1
                                
                                if count % 100 == 0:
                                    print(f"  Processed {count} {shape} images")
                                    
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            
                print(f"Completed {shape}: {count} images")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Loaded {len(X)} images with {X.shape[1]} features each")
        print(f"Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y_encoded
    
    def train_model(self, X, y, test_size=0.2):
        """Train the face shape classification model"""
        print("Training the model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        print("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        # Evaluate on test set
        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, test_pred, target_names=target_names))
        
        return train_accuracy, test_accuracy
    
    def evaluate_model(self):
        """Evaluate model on separate test set"""
        print("Evaluating model on test set...")
        
        # Load test data
        test_features = []
        test_labels = []
        
        test_dir = os.path.join(self.dataset_path, 'testing_set')
        for shape in self.face_shapes:
            shape_dir = os.path.join(test_dir, shape)
            if os.path.exists(shape_dir):
                for img_file in os.listdir(shape_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(shape_dir, img_file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                features = self.extract_features(img)
                                test_features.append(features)
                                test_labels.append(shape)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        if len(test_features) > 0:
            X_test = np.array(test_features)
            y_test_encoded = self.label_encoder.transform(test_labels)
            
            # Predict
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, predictions)
            
            print(f"Test set accuracy: {accuracy:.4f}")
            return accuracy
        else:
            print("No test data found")
            return 0.0
    
    def predict_face_shape(self, image):
        """Predict face shape from image"""
        if isinstance(image, str):
            # Load image from path
            img = cv2.imread(image)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Could not load image")
        else:
            # Assume it's already a numpy array
            img = image
        
        # Extract features
        features = self.extract_features(img)
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        face_shape = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        return face_shape, confidence
    
    def save_model(self, filepath='lightweight_face_shape_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'img_size': self.img_size,
            'face_shapes': self.face_shapes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lightweight_face_shape_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.img_size = model_data['img_size']
            self.face_shapes = model_data['face_shapes']
            
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    # Initialize detector
    detector = LightweightFaceShapeDetector("FaceShape Dataset")
    
    # Load and preprocess data
    X, y = detector.load_and_preprocess_data()
    
    # Train model
    train_acc, test_acc = detector.train_model(X, y)
    
    # Evaluate on separate test set
    final_accuracy = detector.evaluate_model()
    
    # Save model
    detector.save_model()
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {final_accuracy:.4f}")
