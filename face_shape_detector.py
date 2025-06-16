import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

class FaceShapeDetector:
    def __init__(self, dataset_path="FaceShape Dataset"):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.img_size = (224, 224)
        self.face_shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        
    def load_and_preprocess_data(self):
        """Load and preprocess the face shape dataset"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Load training data
        train_dir = os.path.join(self.dataset_path, 'training_set')
        for shape in self.face_shapes:
            shape_dir = os.path.join(train_dir, shape)
            if os.path.exists(shape_dir):
                for img_file in os.listdir(shape_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(shape_dir, img_file)
                        try:
                            # Load and preprocess image
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0  # Normalize
                            
                            images.append(img)
                            labels.append(shape)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded, len(self.face_shapes))
        
        print(f"Loaded {len(X)} images with shape {X.shape}")
        print(f"Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y_categorical
    
    def create_model(self):
        """Create CNN model for face shape classification"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(len(self.face_shapes), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, X, y, epochs=50, validation_split=0.2):
        """Train the face shape detection model"""
        print("Training the model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_face_shape_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")
        
        # Load test data
        test_images = []
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
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0
                            
                            test_images.append(img)
                            test_labels.append(shape)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        X_test = np.array(test_images)
        y_test_encoded = self.label_encoder.transform(test_labels)
        y_test_categorical = keras.utils.to_categorical(y_test_encoded, len(self.face_shapes))
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_categorical)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_accuracy
    
    def predict_face_shape(self, image):
        """Predict face shape from image"""
        if isinstance(image, str):
            # Load image from path
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            img = image
        
        # Preprocess
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        face_shape = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return face_shape, confidence
    
    def save_model(self, filepath='face_shape_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='face_shape_model.h5'):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Initialize detector
    detector = FaceShapeDetector("FaceShape Dataset")
    
    # Load and preprocess data
    X, y = detector.load_and_preprocess_data()
    
    # Create model
    model = detector.create_model()
    print(model.summary())
    
    # Train model
    history = detector.train_model(X, y, epochs=50)
    
    # Evaluate model
    accuracy = detector.evaluate_model()
    
    # Save model
    detector.save_model()
    
    print(f"Training completed! Final test accuracy: {accuracy:.4f}")
