"""
Simplified Advanced Face Shape Trainer
Fixed version without callback pickle issues
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, applications, optimizers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration
print("🎮 Configuring GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU found: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"❌ GPU error: {e}")
else:
    print("⚠️ No GPU detected - using CPU")

class SimpleFaceShapeTrainer:
    def __init__(self):
        self.classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.img_size = (224, 224)
        self.batch_size = 16
        self.model = None
        
    def load_dataset(self, dataset_path="FaceShape Dataset"):
        """Load and preprocess the dataset"""
        print("📊 Loading dataset...")
        
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.classes):
            train_path = os.path.join(dataset_path, "training_set", class_name)
            
            if not os.path.exists(train_path):
                print(f"❌ Path not found: {train_path}")
                continue
                
            print(f"📁 Loading {class_name} images...")
            count = 0
            
            for filename in os.listdir(train_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(train_path, filename)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        
                        X.append(img)
                        y.append(class_idx)
                        count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error loading {filename}: {e}")
                        
            print(f"✅ {class_name}: {count} images loaded")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"🎯 Dataset loaded: {len(X)} images, {X.shape}")
        return X, y
    
    def create_model(self):
        """Create EfficientNetB0-based model"""
        print("🏗️ Creating model...")
        
        # Input layer
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Base model
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )(x)
        
        # Custom head
        x = layers.GlobalAveragePooling2D()(base_model)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.classes), activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully!")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        self.model = model
        return model
    
    def train_simple(self, X, y, epochs=30):
        """Simple training without complex callbacks"""
        print("🚀 Starting training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Training: {len(X_train)}, Validation: {len(X_val)}")
        
        # Simple callbacks (no pickling issues)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X, y):
        """Evaluate the trained model"""
        print("📊 Evaluating model...")
        
        # Split for test set
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Predictions
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Metrics
        accuracy = np.mean(y_pred == y_test)
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        return accuracy
    
    def save_model(self, filename="simple_advanced_model.h5"):
        """Save the trained model"""
        if self.model:
            self.model.save(filename)
            print(f"✅ Model saved as {filename}")
        else:
            print("❌ No model to save!")

def main():
    """Main training function"""
    print("🎭 Simple Advanced Face Shape Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SimpleFaceShapeTrainer()
    
    # Load dataset
    X, y = trainer.load_dataset()
    
    if len(X) == 0:
        print("❌ No data loaded! Check your dataset path.")
        return
    
    # Create model
    model = trainer.create_model()
    
    # Train
    history = trainer.train_simple(X, y, epochs=30)
    
    # Evaluate
    accuracy = trainer.evaluate_model(X, y)
    
    # Save model
    trainer.save_model("simple_advanced_face_model.h5")
    
    print(f"\n🎉 Training completed!")
    print(f"🎯 Final accuracy: {accuracy*100:.2f}%")
    print(f"💾 Model saved as: simple_advanced_face_model.h5")

if __name__ == "__main__":
    main()
