"""
Simple Streamlit web interface for face shape detection
Works with the lightweight model (no TensorFlow required)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from lightweight_detector import LightweightFaceShapeDetector
import os
import base64

# Page configuration
st.set_page_config(
    page_title="Face Shape Detector & Style Recommender",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
    }
    .result-box {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #F8F9FA;
    }
    .recommendation-item {
        background-color: #E8F4FD;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitFaceShapeApp:
    def __init__(self):
        self.face_cascade = None
        self.shape_detector = None
        self.model_loaded = False
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize face detection and shape classification models"""
        try:
            # Load face detection cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Load face shape detector
            self.shape_detector = LightweightFaceShapeDetector()
            if os.path.exists('lightweight_face_shape_model.pkl'):
                self.model_loaded = self.shape_detector.load_model('lightweight_face_shape_model.pkl')
                if self.model_loaded:
                    st.sidebar.success("✅ Model loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load model")
            else:
                st.sidebar.warning("⚠️ Model not found. Please train first!")
                
        except Exception as e:
            st.sidebar.error(f"❌ Initialization error: {e}")
    
    def detect_faces(self, image):
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    
    def get_detailed_recommendations(self, face_shape):
        """Get detailed style recommendations with descriptions"""
        recommendations = {
            'Heart': {
                'description': '💖 Heart-shaped faces have a wider forehead and narrower chin, creating a beautiful heart silhouette.',
                'characteristics': [
                    'Wider forehead and temples',
                    'High cheekbones',
                    'Narrow chin and jawline',
                    'Face is widest at the forehead'
                ],
                'hair': {
                    'best': [
                        '✨ Side-swept bangs to balance the forehead',
                        '🌊 Long layers that add volume at the chin',
                        '💇 Chin-length bob with soft, face-framing layers',
                        '🎀 Soft, wispy bangs',
                        '📐 Asymmetrical cuts'
                    ],
                    'avoid': [
                        '❌ Center parts that emphasize forehead width',
                        '❌ Short, choppy layers',
                        '❌ Severe, blunt cuts'
                    ]
                },
                'beard': {
                    'best': [
                        '🧔 Light stubble to add definition to the jawline',
                        '🎯 Full goatee to balance the narrow chin',
                        '✨ Soul patch for subtle enhancement',
                        '📐 Angular beard styles'
                    ],
                    'avoid': [
                        '❌ Full beards that hide the natural jawline',
                        '❌ Very thick mustaches'
                    ]
                }
            },
            'Oval': {
                'description': '🥚 Oval faces are perfectly balanced and proportioned - you can wear almost any style!',
                'characteristics': [
                    'Length is about 1.5 times the width',
                    'Forehead is slightly wider than chin',
                    'Well-balanced proportions',
                    'Soft, rounded hairline'
                ],
                'hair': {
                    'best': [
                        '🌟 Almost all hairstyles work beautifully',
                        '💇 Shoulder-length cuts with layers',
                        '✂️ Pixie cuts for a bold, chic look',
                        '🌊 Long, flowing waves',
                        '🎭 Dramatic updos',
                        '📏 Blunt cuts and bobs'
                    ],
                    'avoid': [
                        '⚠️ Very few restrictions - experiment freely!'
                    ]
                },
                'beard': {
                    'best': [
                        '🧔 Full beard for a mature, distinguished look',
                        '✨ 5 o\'clock shadow for casual sophistication',
                        '👨 Classic mustache styles',
                        '🎯 Goatee variations',
                        '😊 Clean shaven to highlight natural features'
                    ],
                    'avoid': [
                        '⚠️ Most styles work well - choose based on preference!'
                    ]
                }
            },
            'Round': {
                'description': '🌙 Round faces have soft, curved lines with similar width and length measurements.',
                'characteristics': [
                    'Width and length are nearly equal',
                    'Soft, curved jawline',
                    'Full cheeks',
                    'Rounded hairline'
                ],
                'hair': {
                    'best': [
                        '📏 Layered cuts with height at the crown',
                        '📐 Side parts to create asymmetry',
                        '✨ Angular styles that add definition',
                        '🔺 Long, straight styles to elongate',
                        '💇 High ponytails and updos'
                    ],
                    'avoid': [
                        '❌ Blunt, chin-length cuts',
                        '❌ Center parts',
                        '❌ Very short, rounded styles'
                    ]
                },
                'beard': {
                    'best': [
                        '📐 Angular beard styles to add definition',
                        '🎯 Goatee to elongate the face',
                        '✨ Extended goatee for sharp lines',
                        '🔺 Pointed beard styles'
                    ],
                    'avoid': [
                        '❌ Round, full beards',
                        '❌ Circular mustaches'
                    ]
                }
            },
            'Square': {
                'description': '⬛ Square faces have strong, angular features with a prominent jawline.',
                'characteristics': [
                    'Strong, angular jawline',
                    'Wide forehead',
                    'Face width equals face length',
                    'Squared hairline'
                ],
                'hair': {
                    'best': [
                        '🌊 Soft layers to reduce angularity',
                        '💫 Side-swept styles',
                        '🌀 Wavy textures for softness',
                        '📐 Asymmetrical cuts',
                        '✨ Face-framing layers'
                    ],
                    'avoid': [
                        '❌ Blunt cuts that emphasize angles',
                        '❌ Severe, geometric styles',
                        '❌ Center parts with straight hair'
                    ]
                },
                'beard': {
                    'best': [
                        '🔄 Rounded beard styles to soften angles',
                        '⭕ Circle beard for balance',
                        '✨ Light stubble for subtle softening',
                        '🌙 Curved mustache styles'
                    ],
                    'avoid': [
                        '❌ Sharp, angular beards',
                        '❌ Square-shaped facial hair'
                    ]
                }
            },
            'Oblong': {
                'description': '📏 Oblong faces are longer than they are wide, with elegant proportions.',
                'characteristics': [
                    'Face is longer than it is wide',
                    'High forehead',
                    'Long, straight cheek line',
                    'Narrow chin'
                ],
                'hair': {
                    'best': [
                        '💇 Bangs to shorten the face',
                        '📏 Layered bob for width',
                        '🌊 Wide, voluminous styles',
                        '🔄 Curly or wavy textures',
                        '📐 Side-swept styles with volume'
                    ],
                    'avoid': [
                        '❌ Long, straight styles that elongate',
                        '❌ Center parts',
                        '❌ Very short cuts'
                    ]
                },
                'beard': {
                    'best': [
                        '🧔 Full beard to add width',
                        '📐 Mutton chops for vintage style',
                        '📏 Wide mustache for balance',
                        '🔄 Horizontal beard lines'
                    ],
                    'avoid': [
                        '❌ Goatees that elongate further',
                        '❌ Very thin mustaches'
                    ]
                }
            }
        }
        
        return recommendations.get(face_shape, {
            'description': '🤔 Unknown face shape. Consider consulting with a professional stylist.',
            'characteristics': ['Professional analysis recommended'],
            'hair': {
                'best': ['Professional consultation recommended'],
                'avoid': ['Consult with a stylist']
            },
            'beard': {
                'best': ['Professional consultation recommended'],
                'avoid': ['Consult with a stylist']
            }
        })
    
    def process_image(self, image):
        """Process uploaded image for face shape detection"""
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please train the model first.")
            return None
        
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB image
            img_rgb = img_array
        else:
            st.error("❌ Please upload a color image (RGB)")
            return None
        
        # Detect faces
        faces = self.detect_faces(img_rgb)
        
        if len(faces) == 0:
            st.warning("⚠️ No faces detected in the image. Please try another image with a clear face.")
            return None
        
        results = []
        annotated_image = img_rgb.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face
            face_img = self.extract_face(img_rgb, (x, y, w, h))
            
            if face_img.size > 0:
                try:
                    # Predict face shape
                    face_shape, confidence = self.shape_detector.predict_face_shape(face_img)
                    
                    # Get detailed recommendations
                    recommendations = self.get_detailed_recommendations(face_shape)
                    
                    result = {
                        'face_number': i + 1,
                        'face_shape': face_shape,
                        'confidence': confidence,
                        'recommendations': recommendations,
                        'face_coords': (x, y, w, h),
                        'face_image': face_img
                    }
                    results.append(result)
                    
                    # Draw annotation on image
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(annotated_image, f"{face_shape}: {confidence:.2f}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                except Exception as e:
                    st.error(f"❌ Error processing face {i+1}: {e}")
        
        return results, annotated_image

def main():
    # Initialize app
    app = StreamlitFaceShapeApp()
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Face Shape Detector & Style Recommender</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Train Model", "📸 Detect Face Shape", "ℹ️ Face Shape Guide", "📞 About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🔧 Train Face Shape Detection Model</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📊 Dataset Information
            - **Total Images**: ~4000 images across 5 face shapes
            - **Categories**: Heart, Oval, Round, Square, Oblong  
            - **Model Type**: Random Forest Classifier
            - **Features**: Pixel intensities + geometric measurements
            """)
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize detector
                        status_text.text("🔧 Initializing detector...")
                        progress_bar.progress(20)
                        
                        detector = LightweightFaceShapeDetector("FaceShape Dataset")
                        
                        # Load data
                        status_text.text("📊 Loading and preprocessing data...")
                        progress_bar.progress(40)
                        X, y = detector.load_and_preprocess_data()
                        
                        # Train model
                        status_text.text("🚀 Training model...")
                        progress_bar.progress(70)
                        train_acc, test_acc = detector.train_model(X, y)
                        
                        # Evaluate
                        status_text.text("📈 Evaluating model...")
                        progress_bar.progress(90)
                        final_accuracy = detector.evaluate_model()
                        
                        # Save model
                        status_text.text("💾 Saving model...")
                        detector.save_model()
                        progress_bar.progress(100)
                        
                        status_text.text("✅ Training completed!")
                        
                        st.success(f"🎉 Model trained successfully!")
                        st.info(f"📊 Final test accuracy: {final_accuracy:.2%}")
                        
                        # Refresh the page to load the new model
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
        
        with col2:
            st.markdown("""
            ### 🎯 Model Performance
            - **Expected Accuracy**: 60-80%
            - **Training Time**: 2-5 minutes
            - **Model Size**: ~2MB
            - **Requirements**: scikit-learn, OpenCV
            """)
    
    with tab2:
        st.markdown('<h2 class="sub-header">📸 Face Shape Detection & Style Recommendations</h2>', unsafe_allow_html=True)
        
        if not app.model_loaded:
            st.warning("⚠️ Model not loaded. Please train the model first in the 'Train Model' tab.")
            return
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo with visible face(s)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Detect Face Shape", type="primary", use_container_width=True):
                    with st.spinner("Analyzing face shape..."):
                        results = app.process_image(image)
                        
                        if results:
                            detection_results, annotated_img = results
                            
                            # Display annotated image
                            st.subheader("🎯 Detection Results")
                            st.image(annotated_img, caption="Detected Faces", use_column_width=True)
                            
                            # Store results in session state for display in col2
                            st.session_state.detection_results = detection_results
            
            with col2:
                if hasattr(st.session_state, 'detection_results'):
                    st.subheader("💎 Style Recommendations")
                    
                    for result in st.session_state.detection_results:
                        recommendations = result['recommendations']
                        
                        # Face shape result
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>👤 Face {result['face_number']}: {result['face_shape']}</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                            <p>{recommendations['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Characteristics
                        with st.expander("🔍 Face Shape Characteristics"):
                            for char in recommendations['characteristics']:
                                st.write(f"• {char}")
                        
                        # Hair recommendations
                        with st.expander("💇 Hairstyle Recommendations"):
                            st.markdown("**✨ Best Styles:**")
                            for style in recommendations['hair']['best']:
                                st.markdown(f'<div class="recommendation-item">{style}</div>', unsafe_allow_html=True)
                            
                            if recommendations['hair']['avoid']:
                                st.markdown("**⚠️ Styles to Avoid:**")
                                for style in recommendations['hair']['avoid']:
                                    st.write(f"• {style}")
                        
                        # Beard recommendations
                        with st.expander("🧔 Beard Style Recommendations"):
                            st.markdown("**✨ Best Styles:**")
                            for style in recommendations['beard']['best']:
                                st.markdown(f'<div class="recommendation-item">{style}</div>', unsafe_allow_html=True)
                            
                            if recommendations['beard']['avoid']:
                                st.markdown("**⚠️ Styles to Avoid:**")
                                for style in recommendations['beard']['avoid']:
                                    st.write(f"• {style}")
                        
                        st.markdown("---")
    
    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ Face Shape Guide</h2>', unsafe_allow_html=True)
        
        # Create visual guide for face shapes
        col1, col2 = st.columns(2)
        
        shapes_info = {
            '💖 Heart': 'Wider forehead, narrow chin, high cheekbones',
            '🥚 Oval': 'Balanced proportions, slightly longer than wide',
            '🌙 Round': 'Equal width and height, soft curves',
            '⬛ Square': 'Strong jawline, angular features',
            '📏 Oblong': 'Longer than wide, high forehead'
        }
        
        for i, (shape, description) in enumerate(shapes_info.items()):
            if i % 2 == 0:
                with col1:
                    st.markdown(f"""
                    ### {shape}
                    {description}
                    """)
            else:
                with col2:
                    st.markdown(f"""
                    ### {shape}
                    {description}
                    """)
        
        st.markdown("---")
        st.markdown("""
        ### 📐 How to Determine Your Face Shape
        1. **Measure your face**: Width of forehead, cheekbones, jawline, and face length
        2. **Compare proportions**: Which measurement is largest?
        3. **Observe angles**: Are features soft and curved or sharp and angular?
        4. **Use our detector**: Upload a photo for automatic detection!
        """)
    
    with tab4:
        st.markdown('<h2 class="sub-header">📞 About This Application</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Features
            - **AI-Powered Detection**: Machine learning face shape classification
            - **Style Recommendations**: Personalized hair and beard suggestions
            - **Real-time Processing**: Fast detection and analysis
            - **User-Friendly**: Simple web interface
            - **Offline Capable**: No internet required after setup
            """)
            
            st.markdown("""
            ### 🛠️ Technology Stack
            - **Frontend**: Streamlit
            - **Computer Vision**: OpenCV
            - **Machine Learning**: scikit-learn (Random Forest)
            - **Image Processing**: PIL/Pillow
            - **Data Processing**: NumPy
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Model Information
            - **Dataset Size**: ~4000 images
            - **Categories**: 5 face shapes
            - **Algorithm**: Random Forest Classifier
            - **Features**: 4100+ dimensional feature vector
            - **Accuracy**: 60-80% (varies by image quality)
            """)
            
            st.markdown("""
            ### 💡 Tips for Best Results
            - Use clear, well-lit photos
            - Ensure face is visible and unobstructed
            - Avoid extreme angles or expressions
            - High-resolution images work better
            - Remove accessories that hide face shape
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🤝 Support & Feedback
        This application uses traditional machine learning for face shape detection. 
        For better accuracy, consider using deep learning models with larger datasets.
        
        **Made with ❤️ using Python and Streamlit**
        """)

if __name__ == "__main__":
    main()
