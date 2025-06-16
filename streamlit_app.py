import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from face_shape_detector import FaceShapeDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="Face Shape Detector & Style Recommender",
    page_icon="👤",
    layout="wide"
)

class StreamlitFaceShapeDetector:
    def __init__(self):
        self.face_cascade = None
        self.shape_detector = None
        self.model_loaded = False
        
    @st.cache_resource
    def load_model(_self):
        """Load the face shape detection model and face cascade"""
        try:
            # Load face detection cascade
            _self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load face shape detector
            _self.shape_detector = FaceShapeDetector()
            _self.shape_detector.load_model('face_shape_model.h5')
            _self.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def detect_faces(self, image):
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_face(self, image, face_coords):
        """Extract face region from image"""
        x, y, w, h = face_coords
        padding = int(0.2 * min(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_img = image[y1:y2, x1:x2]
        return face_img
    
    def get_style_recommendations(self, face_shape):
        """Get detailed style recommendations"""
        recommendations = {
            'Heart': {
                'description': 'Heart-shaped faces have a wider forehead and narrower chin.',
                'hair': [
                    'Side-swept bangs to balance the forehead',
                    'Long layers that add volume at the chin',
                    'Chin-length bob with soft layers',
                    'Avoid: Center parts, short choppy layers'
                ],
                'beard': [
                    'Light stubble to add definition',
                    'Full goatee to balance the chin',
                    'Soul patch for subtle enhancement',
                    'Avoid: Full beards that hide the jawline'
                ]
            },
            'Oval': {
                'description': 'Oval faces are well-balanced and can suit most styles.',
                'hair': [
                    'Most hairstyles work well',
                    'Shoulder-length cuts with layers',
                    'Pixie cuts for a bold look',
                    'Long waves for elegance'
                ],
                'beard': [
                    'Full beard for a mature look',
                    'Stubble for casual style',
                    'Mustache for vintage appeal',
                    'Clean shaven to highlight features'
                ]
            },
            'Round': {
                'description': 'Round faces benefit from angular and elongating styles.',
                'hair': [
                    'Layered cuts with height at the crown',
                    'Side parts to create asymmetry',
                    'Angular styles that add definition',
                    'Avoid: Blunt cuts, center parts'
                ],
                'beard': [
                    'Angular beard styles',
                    'Goatee to elongate the face',
                    'Extended goatee for definition',
                    'Avoid: Round, full beards'
                ]
            },
            'Square': {
                'description': 'Square faces have strong jawlines that benefit from softening.',
                'hair': [
                    'Soft layers to reduce angularity',
                    'Side-swept styles',
                    'Wavy textures for softness',
                    'Avoid: Blunt cuts, severe angles'
                ],
                'beard': [
                    'Rounded beard styles',
                    'Circle beard for balance',
                    'Light stubble for subtle softening',
                    'Avoid: Sharp, angular beards'
                ]
            },
            'Oblong': {
                'description': 'Oblong faces are longer and benefit from width-adding styles.',
                'hair': [
                    'Bangs to shorten the face',
                    'Layered bob for width',
                    'Wide, voluminous styles',
                    'Avoid: Long, straight styles'
                ],
                'beard': [
                    'Full beard to add width',
                    'Mutton chops for vintage style',
                    'Wide mustache for balance',
                    'Avoid: Goatees that elongate'
                ]
            }
        }
        
        return recommendations.get(face_shape, {
            'description': 'Consult with a professional stylist.',
            'hair': ['Professional consultation recommended'],
            'beard': ['Professional consultation recommended']
        })

def main():
    st.title("🎭 Face Shape Detector & Style Recommender")
    st.markdown("---")
    
    # Initialize detector
    detector = StreamlitFaceShapeDetector()
    
    # Sidebar
    st.sidebar.title("📋 Instructions")
    st.sidebar.markdown("""
    1. **Train Model**: First train the face shape detection model
    2. **Upload Image**: Upload a photo to detect face shape
    3. **Get Recommendations**: Receive personalized style suggestions
    4. **Live Camera**: Use your camera for real-time detection
    """)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Train Model", "📸 Upload Image", "📹 Live Camera", "ℹ️ About"])
    
    with tab1:
        st.header("Train Face Shape Detection Model")
        st.markdown("First, we need to train the model on your dataset.")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model... This may take several minutes."):
                try:
                    # Initialize and train model
                    face_detector = FaceShapeDetector("FaceShape Dataset")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load data
                    status_text.text("Loading and preprocessing data...")
                    progress_bar.progress(20)
                    X, y = face_detector.load_and_preprocess_data()
                    
                    # Create model
                    status_text.text("Creating model architecture...")
                    progress_bar.progress(40)
                    model = face_detector.create_model()
                    
                    # Train model
                    status_text.text("Training model...")
                    progress_bar.progress(60)
                    history = face_detector.train_model(X, y, epochs=30)
                    
                    # Evaluate
                    status_text.text("Evaluating model...")
                    progress_bar.progress(80)
                    accuracy = face_detector.evaluate_model()
                    
                    # Save model
                    status_text.text("Saving model...")
                    progress_bar.progress(90)
                    face_detector.save_model()
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    st.success(f"🎉 Model trained successfully! Test accuracy: {accuracy:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
    
    with tab2:
        st.header("Upload Image for Face Shape Detection")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Detect Face Shape", type="primary"):
                if detector.load_model():
                    try:
                        # Convert PIL image to OpenCV format
                        img_array = np.array(image)
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Detect faces
                        faces = detector.detect_faces(img_bgr)
                        
                        if len(faces) == 0:
                            st.warning("⚠️ No faces detected in the image. Please try another image.")
                        else:
                            for i, (x, y, w, h) in enumerate(faces):
                                # Extract face
                                face_img = detector.extract_face(img_bgr, (x, y, w, h))
                                
                                if face_img.size > 0:
                                    # Convert to RGB for prediction
                                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                    
                                    # Predict face shape
                                    face_shape, confidence = detector.shape_detector.predict_face_shape(face_rgb)
                                    
                                    # Display results
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader(f"Face {i+1} Results")
                                        st.metric("Face Shape", face_shape)
                                        st.metric("Confidence", f"{confidence:.2%}")
                                        
                                        # Display extracted face
                                        face_pil = Image.fromarray(face_rgb)
                                        st.image(face_pil, caption="Detected Face", width=200)
                                    
                                    with col2:
                                        # Get recommendations
                                        recommendations = detector.get_style_recommendations(face_shape)
                                        
                                        st.subheader("Style Recommendations")
                                        st.write(recommendations['description'])
                                        
                                        st.subheader("💇 Hairstyle Suggestions")
                                        for hair_tip in recommendations['hair']:
                                            st.write(f"• {hair_tip}")
                                        
                                        st.subheader("🧔 Beard Style Suggestions")
                                        for beard_tip in recommendations['beard']:
                                            st.write(f"• {beard_tip}")
                                    
                                    st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Detection failed: {e}")
                else:
                    st.error("❌ Model not loaded. Please train the model first.")
    
    with tab3:
        st.header("Live Camera Detection")
        st.markdown("**Note**: This feature requires a webcam and may not work in all environments.")
        
        if st.button("🎥 Start Camera"):
            st.info("Camera feature would be available in a local environment with proper webcam access.")
            st.markdown("""
            To use the camera feature locally:
            1. Run `python camera_detector.py` in your terminal
            2. Choose option 1 for real-time detection
            3. Press 'q' to quit
            """)
    
    with tab4:
        st.header("About Face Shape Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Face Shape Categories")
            st.markdown("""
            - **Heart**: Wider forehead, narrower chin
            - **Oval**: Well-balanced proportions
            - **Round**: Similar width and height
            - **Square**: Strong, angular jawline
            - **Oblong**: Longer than it is wide
            """)
        
        with col2:
            st.subheader("🎯 How It Works")
            st.markdown("""
            1. **Face Detection**: Uses OpenCV to locate faces
            2. **Shape Classification**: CNN model predicts face shape
            3. **Style Recommendation**: Provides personalized suggestions
            4. **Real-time Processing**: Works with camera or uploaded images
            """)
        
        st.subheader("📊 Model Architecture")
        st.markdown("""
        - **Input**: 224x224 RGB images
        - **Architecture**: Convolutional Neural Network (CNN)
        - **Layers**: 4 Conv blocks + Dense layers
        - **Output**: 5 face shape classes with confidence scores
        """)

if __name__ == "__main__":
    main()
