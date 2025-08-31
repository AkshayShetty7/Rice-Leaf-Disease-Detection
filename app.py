import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from prediction import RiceDiseasePredictor
import time

# Page configuration
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2E8B57;
    }
    
    .disease-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    
    .healthy-result {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    
    .disease-result {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Disease information database
DISEASE_INFO = {
    'Bacterial Leaf Blight': {
        'description': 'A bacterial disease that causes yellowing and drying of rice leaves.',
        'symptoms': ['Water-soaked lesions', 'Yellow to brown streaks', 'Leaf tips turning yellow'],
        'treatment': ['Use resistant varieties', 'Apply copper-based bactericides', 'Improve field drainage'],
        'severity': 'High'
    },
    'Brown Spot': {
        'description': 'A fungal disease causing brown spots on rice leaves and grains.',
        'symptoms': ['Circular brown spots', 'Dark brown borders', 'Yellowing of leaves'],
        'treatment': ['Apply fungicides', 'Use certified seeds', 'Maintain proper spacing'],
        'severity': 'Medium'
    },
    'Healthy Rice Leaf': {
        'description': 'Healthy rice leaves with no visible disease symptoms.',
        'symptoms': ['Green color', 'No spots or lesions', 'Normal growth pattern'],
        'treatment': ['Continue good practices', 'Regular monitoring', 'Preventive measures'],
        'severity': 'None'
    },
    'Leaf Blast': {
        'description': 'A fungal disease causing diamond-shaped lesions on rice leaves.',
        'symptoms': ['Diamond-shaped spots', 'Gray centers with brown borders', 'Leaf burning'],
        'treatment': ['Apply fungicides', 'Use resistant varieties', 'Avoid excess nitrogen'],
        'severity': 'High'
    },
    'Leaf Scald': {
        'description': 'A fungal disease causing scalded appearance on rice leaves.',
        'symptoms': ['Large irregular spots', 'Zonate patterns', 'Premature leaf death'],
        'treatment': ['Apply fungicides', 'Improve air circulation', 'Remove infected debris'],
        'severity': 'Medium'
    },
    'Sheath Blight': {
        'description': 'A fungal disease affecting rice sheaths and lower leaves.',
        'symptoms': ['Oval spots on sheaths', 'Water-soaked lesions', 'Irregular brown patches'],
        'treatment': ['Apply fungicides', 'Reduce plant density', 'Manage water levels'],
        'severity': 'High'
    }
}

@st.cache_resource
def load_model():
    """Load the rice disease prediction model"""
    try:
        predictor = RiceDiseasePredictor('rice_disease_model_2.h5')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def display_disease_info(disease_name, confidence):
    """Display detailed information about the predicted disease"""
    info = DISEASE_INFO.get(disease_name, {})
    
    if disease_name == 'Healthy Rice Leaf':
        st.markdown(f"""
        <div class="healthy-result">
            <h3>✅ {disease_name}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Status:</strong> Your rice leaf appears healthy!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        severity_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
        color = severity_color.get(info.get('severity', 'Medium'), "#ffc107")
        
        st.markdown(f"""
        <div class="disease-result">
            <h3>⚠️ {disease_name}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Severity:</strong> <span style="color: {color}; font-weight: bold;">{info.get('severity', 'Unknown')}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    if info:
        with st.expander("📋 Detailed Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Description:**")
                st.write(info['description'])
                
                st.markdown("**Symptoms:**")
                for symptom in info['symptoms']:
                    st.write(f"• {symptom}")
            
            with col2:
                st.markdown("**Treatment Recommendations:**")
                for treatment in info['treatment']:
                    st.write(f"• {treatment}")

def create_confidence_chart(predictions, class_names):
    """Create a confidence chart for all classes"""
    df = pd.DataFrame({
        'Disease': class_names,
        'Confidence': predictions * 100
    })
    
    # Sort by confidence
    df = df.sort_values('Confidence', ascending=True)
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Disease', 
        orientation='h',
        title='Prediction Confidence for All Classes',
        color='Confidence',
        color_continuous_scale='RdYlGn',
        text='Confidence'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Confidence (%)",
        yaxis_title="Disease Type"
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Leaf Disease Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload an image of a rice leaf to detect potential diseases using AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        predictor = load_model()
    
    if predictor is None:
        st.error("Failed to load the prediction model. Please check if the model file exists.")
        return
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.info("""
        **Model:** EfficientNet-based CNN
        **Classes:** 6 rice diseases
        **Accuracy:** ~96%+
        **Input Size:** 224×224 pixels
        """)
        
        st.markdown("### 🔍 Detectable Diseases")
        for disease in DISEASE_INFO.keys():
            if disease != 'Healthy Rice Leaf':
                severity = DISEASE_INFO[disease]['severity']
                emoji = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                st.write(f"{emoji} {disease}")
            else:
                st.write(f"✅ {disease}")
        
        st.markdown("### 📝 Instructions")
        st.write("""
        1. Upload a clear image of a rice leaf
        2. Ensure good lighting and focus
        3. Single leaf works best
        4. Wait for AI analysis
        5. Review results and recommendations
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a rice leaf image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a rice leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Rice Leaf", use_column_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"• **Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"• **Format:** {image.format}")
            st.write(f"• **Mode:** {image.mode}")
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Prediction button
            if st.button("🚀 Analyze Rice Leaf", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait"):
                    # Add progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        # Make prediction
                        predicted_class, confidence, all_predictions = predictor.predict_image(image)
                        
                        # Clear progress bar
                        progress_bar.empty()
                        
                        # Display main result
                        display_disease_info(predicted_class, confidence)
                        
                        # Show confidence chart
                        st.markdown("### 📊 Detailed Analysis")
                        fig = create_confidence_chart(all_predictions, predictor.class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        with st.expander("🔍 Analysis Insights"):
                            st.write("**Top 3 Predictions:**")
                            top_3_indices = np.argsort(all_predictions)[-3:][::-1]
                            for i, idx in enumerate(top_3_indices):
                                confidence_pct = all_predictions[idx] * 100
                                st.write(f"{i+1}. {predictor.class_names[idx]}: {confidence_pct:.2f}%")
                            
                            st.write("\n**Confidence Level:**")
                            if confidence > 0.9:
                                st.success("Very High Confidence - Results are highly reliable")
                            elif confidence > 0.7:
                                st.info("Good Confidence - Results are reliable")
                            elif confidence > 0.5:
                                st.warning("Moderate Confidence - Consider retaking image")
                            else:
                                st.error("Low Confidence - Please upload a clearer image")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.write("Please try uploading a different image or check the model file.")
    
    # Additional information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📚 About Rice Diseases</h2>', unsafe_allow_html=True)
    
    # Create tabs for disease information
    tabs = st.tabs([disease for disease in DISEASE_INFO.keys()])
    
    for tab, (disease_name, disease_info) in zip(tabs, DISEASE_INFO.items()):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {disease_info['description']}")
                
                st.write("**Symptoms:**")
                for symptom in disease_info['symptoms']:
                    st.write(f"• {symptom}")
                
                st.write("**Treatment:**")
                for treatment in disease_info['treatment']:
                    st.write(f"• {treatment}")
            
            with col2:
                severity = disease_info['severity']
                if severity == 'High':
                    st.error(f"Severity: {severity}")
                elif severity == 'Medium':
                    st.warning(f"Severity: {severity}")
                elif severity == 'None':
                    st.success(f"Status: Healthy")
                else:
                    st.info(f"Severity: {severity}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌾 Rice Leaf Disease Detection System | Powered by EfficientNet & TensorFlow</p>
        <p>For best results, upload clear, well-lit images of individual rice leaves</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()