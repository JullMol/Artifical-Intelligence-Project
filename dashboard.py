import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gdown
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from PIL import Image
import io
import zipfile

# Page Configuration
st.set_page_config(
    page_title="AI Stroke Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .normal-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .bleeding-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .ischemia-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive Links (Ganti dengan link Anda yang sudah di-share publicly)
GDRIVE_LINKS = {
    'svm': 'https://drive.google.com/uc?id=1ywToTGcjHcXdTG3L3VGI9HcgAMJ5jbeg',
    'knn': 'https://drive.google.com/uc?id=1L7QpCS7qw3Xs0lE4LelKrW00jkL-4YxB',
    'dt':  'https://drive.google.com/uc?id=1Pc_1kbDNDe2nusV-PLuLHM0K4UEpaVAf',
    'xgb': 'https://drive.google.com/uc?id=1O7qCrqOHAUf_MqgZ6OUD6VVVaN6CUAwJ',
    'scaler_standard': 'https://drive.google.com/uc?id=1yaLVtW1g1T92Tc8fj7VirB9mu04sgTOn',
    'scaler_minmax':  'https://drive.google.com/uc?id=1QymHNi-HoVrWZ-mpkf2E2y4CjYKt6JUK',
    'pca': 'https://drive.google.com/uc?id=19eJU4NHkLk1u5BVQBhmiKDB7zmO-D1SA',
    'label_encoder': 'https://drive.google.com/uc?id=1XcKnuyEjIIktjh6A-eIh6dFYtdDGlkqS',
    'extracted_features': 'https://drive.google.com/uc?id=1635XehoTjAvaLJGC6VTsilmDiXEsVeBG'
}

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False

# Helper Functions
@st.cache_data
def download_from_gdrive(url, output_path):
    """Download file from Google Drive"""
    try:
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error downloading: {e}")
        return False

@st.cache_resource
def load_models():
    """Load pre-trained models and scalers. Supports individual model files and two scaler files."""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    expected = {
        'svm': 'svm_model.pkl',
        'knn': 'knn_model.pkl',
        'dt':  'dt_model.pkl',
        'xgb': 'xgboost_model.pkl',
        'scaler_standard': 'scaler_standard.pkl',
        'scaler_minmax':  'scaler_minmax.pkl',
        'pca': 'pca.pkl',
        'label_encoder': 'label_encoder.pkl'
    }

    # Download missing files from GDrive if links provided
    for key, url in GDRIVE_LINKS.items():
        out_path = os.path.join(models_dir, expected.get(key, f"{key}.pkl"))
        if not os.path.exists(out_path) and url and 'YOUR_' not in url:
            try:
                download_from_gdrive(url, out_path)
            except Exception:
                # download_from_gdrive already reports errors via st.error
                pass

    models = {}
    scalers = {}
    pca = None
    label_encoder = None

    # Load each model file if present
    for mname in ['svm', 'knn', 'dt', 'xgb']:
        path = os.path.join(models_dir, expected[mname])
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[mname] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {mname} from {path}: {e}")

    # Load two scaler files (expecting either a dict or a scaler object)
    for s_key in ['scaler_standard', 'scaler_minmax']:
        path = os.path.join(models_dir, expected[s_key])
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    # jika file berisi dict seperti {'hog': StandardScaler(), 'lbp': ...}
                    scalers.update(data)
                else:
                    # single scaler object -> simpan dengan kunci khusus
                    scalers[s_key] = data
            except Exception as e:
                st.warning(f"Failed to load {s_key}: {e}")

    # Load PCA if available
    pca_path = os.path.join(models_dir, expected['pca'])
    if os.path.exists(pca_path):
        try:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load PCA: {e}")

    # Load label encoder
    le_path = os.path.join(models_dir, expected['label_encoder'])
    if os.path.exists(le_path):
        try:
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load label encoder: {e}")

    # If no models found, return None to trigger upload/error in UI
    if not models:
        return None, None, None, None

    return models, scalers, pca, label_encoder

@st.cache_data
def load_sample_dataset():
    """Load sample dataset for overview"""
    sample_dir = 'sample_data'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Load extracted features CSV
    features_path = os.path.join(sample_dir, 'extracted_features')
    
    try:
        df = pd.read_csv(features_path)
        return df
    except FileNotFoundError:
        return None

def extract_features(img_color, img_gray):
    """Extract HOG, LBP, and Color Histogram features"""
    # HOG
    hog_feat = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', 
                   transform_sqrt=True, visualize=False)
    
    # LBP
    radius, n_points = 2, 16
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                                range=(0, n_points + 2))
    lbp_feat = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-6)
    
    # Color Histogram
    hist_feat = []
    for i in range(3):
        h = cv2.calcHist([img_color], [i], None, [32], [0, 256])
        hist_feat.extend(h.flatten())
    hist_feat = np.array(hist_feat) / (np.sum(hist_feat) + 1e-6)
    
    return hog_feat, lbp_feat, hist_feat

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for prediction"""
    img_resized = cv2.resize(image, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_resized, img_gray

def predict_image(image, models, scalers, pca, label_encoder):
    """Make prediction on image"""
    # Preprocess
    img_resized, img_gray = preprocess_image(image)
    
    # Extract features
    hog_feat, lbp_feat, hist_feat = extract_features(img_resized, img_gray)
    
    # Normalize features
    hog_norm = scalers['hog'].transform([hog_feat])
    lbp_norm = scalers['lbp'].transform([lbp_feat])
    hist_norm = scalers['hist'].transform([hist_feat])
    
    # Apply PCA if exists
    if pca is not None:
        hog_pca = pca.transform(hog_norm)
        X_pred = np.hstack([hog_pca, lbp_norm, hist_norm])
    else:
        X_pred = np.hstack([hog_norm, lbp_norm, hist_norm])
    
    # Predictions from all models
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_pred)[0]
        pred_class = label_encoder.inverse_transform([pred])[0]
        proba = model.predict_proba(X_pred)[0]
        confidence = np.max(proba) * 100
        
        predictions[name] = {
            'class': pred_class,
            'confidence': confidence,
            'probabilities': dict(zip(label_encoder.classes_, proba))
        }
    
    return predictions, (hog_feat, lbp_feat, hist_feat)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
    st.title("🧠 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📊 Dataset Overview", "🔬 Feature Extraction", 
         "🎯 Prediction", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Load models button
    if not st.session_state.models_loaded:
        if st.button("📥 Load Pre-trained Models", type="primary"):
            with st.spinner("Loading models..."):
                models, scalers, pca, label_encoder = load_models()
                
                if models is not None:
                    st.session_state.models = models
                    st.session_state.scalers = scalers
                    st.session_state.pca = pca
                    st.session_state.label_encoder = label_encoder
                    st.session_state.class_names = label_encoder.classes_
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded!")
                    st.rerun()
                else:
                    st.error("❌ Models not found. Please upload model files.")
    else:
        st.success("✅ Models Ready")
        if st.button("🔄 Reload Models"):
            st.session_state.models_loaded = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("- [📁 Google Drive Dataset](https://drive.google.com/drive/folders/1arvBtDxdOE8-7caIXArXVTxOoS1vSUgd)")
    st.markdown("- [💻 GitHub Repository](#)")
    
    st.markdown("---")
    st.info("💡 **Tip**: Upload model files (PKL) if not found locally")

# ==================== MAIN CONTENT ====================

# Header
st.markdown('<p class="main-header">🧠 AI Stroke Detection & Classification System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Turkish Ministry of Health Open Dataset Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to AI Stroke Detection System")
        
        st.markdown("""
        ### 🎯 **Key Features**
        
        This advanced AI system leverages **machine learning** and **computer vision** techniques to detect 
        and classify stroke types from CT brain images with high accuracy.
        
        #### 🔍 **What We Detect:**
        - 🔴 **Bleeding** - Hemorrhagic stroke (blood vessel rupture)
        - 🟡 **Ischemia** - Ischemic stroke (blocked blood flow)
        - 🟢 **Normal** - No stroke detected
        
        #### 🧬 **Feature Extraction Methods:**
        1. **HOG (Histogram of Oriented Gradients)** - Captures edges and shapes
        2. **LBP (Local Binary Patterns)** - Analyzes texture patterns
        3. **Color Histogram** - Evaluates color distribution across RGB channels
        
        #### 🤖 **Machine Learning Models:**
        - **SVM** (Support Vector Machine) with RBF kernel
        - **KNN** (K-Nearest Neighbors) with distance weighting
        - **Decision Tree** with optimized hyperparameters
        - **XGBoost** (Extreme Gradient Boosting)
        
        #### 📋 **How to Use:**
        1. **Load Models** - Click the button in sidebar to load pre-trained models
        2. **Explore Dataset** - View sample images and statistics
        3. **Feature Extraction** - See how features are extracted from CT scans
        4. **Make Predictions** - Upload your own CT scan images
        5. **Compare Models** - Analyze performance metrics
        """)
        
        if st.session_state.models_loaded:
            st.success("✅ **System Ready!** You can start making predictions.")
        else:
            st.warning("⚠️ **Please load pre-trained models from the sidebar to get started.**")
    
    with col2:
        st.markdown("### 📊 Quick Statistics")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Size", "2,350 Images", help="Total CT scan images in dataset")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classes", "3 Types", help="Bleeding, Ischemia, Normal")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models", "4 Algorithms", help="SVM, KNN, DT, XGBoost")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Accuracy", "96.8%", help="Achieved by SVM model")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🏥 Clinical Impact")
        st.info("""
        **Early detection** of stroke is critical:
        - ⏰ Time = Brain tissue
        - 🎯 Accurate classification guides treatment
        - 🚑 Faster triage in emergency rooms
        """)

# ==================== DATASET OVERVIEW PAGE ====================
elif page == "📊 Dataset Overview":
    st.markdown('<p class="sub-header">Dataset Overview & Statistics</p>', unsafe_allow_html=True)
    
    # Dataset Information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Images", "2,350")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Image Size", "128×128 px")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classes", "3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Class Distribution
    st.markdown("### 📊 Class Distribution")
    
    # Sample data (ganti dengan data real dari CSV jika ada)
    class_data = {
        'Class': ['Bleeding', 'Ischemia', 'Normal'],
        'Count': [750, 800, 800],
        'Percentage': [31.9, 34.0, 34.0]
    }
    df_classes = pd.DataFrame(class_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#ff6b6b', '#feca57', '#48dbfb']
        bars = ax.bar(df_classes['Class'], df_classes['Count'], color=colors, edgecolor='black', linewidth=2)
        ax.set_xlabel('Stroke Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Image Distribution by Class', fontsize=14, fontweight='bold')
        
        for bar, count in zip(bars, df_classes['Count']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Pie Chart")
        fig, ax = plt.subplots(figsize=(8, 8))
        explode = (0.05, 0.05, 0.05)
        ax.pie(df_classes['Count'], labels=df_classes['Class'], autopct='%1.1f%%', 
               colors=colors, startangle=90, explode=explode,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Class Distribution Percentage', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Data Table
    st.markdown("### 📋 Detailed Statistics")
    st.dataframe(df_classes.style.highlight_max(axis=0, subset=['Count']), use_container_width=True)
    
    st.markdown("---")
    
    # Sample Images Section
    st.markdown("### 🖼️ Sample Images from Dataset")
    st.info("**Note**: These are representative samples from the full dataset stored in Google Drive.")
    
    # Upload sample images option
    st.markdown("#### Upload Sample Images (Optional)")
    uploaded_samples = st.file_uploader("Upload sample images for each class", 
                                       type=['png', 'jpg', 'jpeg'], 
                                       accept_multiple_files=True)
    
    if uploaded_samples:
        cols = st.columns(min(3, len(uploaded_samples)))
        for idx, uploaded in enumerate(uploaded_samples[:9]):
            with cols[idx % 3]:
                image = Image.open(uploaded)
                st.image(image, caption=f"Sample {idx+1}", use_container_width=True)
    else:
        st.warning("Upload sample images or access the full dataset from Google Drive link in sidebar.")
    
    st.markdown("---")
    
    # Feature Statistics
    st.markdown("### 📈 Feature Extraction Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**HOG Features**")
        st.write("- Orientations: 9")
        st.write("- Pixels per cell: 8×8")
        st.write("- Cells per block: 2×2")
        st.write("- Total features: ~8,100")
    
    with col2:
        st.markdown("**LBP Features**")
        st.write("- Radius: 2")
        st.write("- Points: 16")
        st.write("- Method: Uniform")
        st.write("- Total features: 18")
    
    with col3:
        st.markdown("**Color Histogram**")
        st.write("- Channels: RGB (3)")
        st.write("- Bins per channel: 32")
        st.write("- Total features: 96")
    
    st.info("💡 **After PCA**: Features are reduced to 200 dimensions for optimal performance while retaining 95%+ variance.")

# ==================== FEATURE EXTRACTION PAGE ====================
elif page == "🔬 Feature Extraction":
    st.markdown('<p class="sub-header">Feature Extraction Visualization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔍 Understanding Feature Extraction
    
    This system uses **three complementary feature extraction methods** to capture different aspects of CT scan images:
    
    1. **HOG (Histogram of Oriented Gradients)** - Detects edges and structural patterns
    2. **LBP (Local Binary Patterns)** - Captures texture information
    3. **Color Histogram** - Analyzes intensity distribution across color channels
    """)
    
    st.markdown("---")
    
    # Upload image for demonstration
    st.markdown("### 📤 Upload a CT Scan Image")
    uploaded_file = st.file_uploader("Choose a CT scan image to visualize feature extraction", 
                                     type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Preprocess
        img_resized, img_gray = preprocess_image(image)
        
        # Display original image
        st.markdown("#### 🖼️ Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 
                    caption="Preprocessed CT Scan (128×128)", 
                    use_container_width=True)
        
        st.markdown("---")
        
        # Extract features
        with st.spinner("🔄 Extracting features..."):
            hog_feat, lbp_feat, hist_feat = extract_features(img_resized, img_gray)
        
        # Feature Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔲 HOG Features", "🔳 LBP Features", "🎨 Color Histogram", "📊 Summary"])
        
        with tab1:
            st.markdown("### Histogram of Oriented Gradients (HOG)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 HOG Visualization")
                from skimage import exposure
                _, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(hog_image, cmap='gray')
                ax.set_title('HOG Feature Visualization', fontsize=14, fontweight='bold')
                ax.axis('off')
                st.pyplot(fig)
                
                st.info(f"**Total HOG Features**: {len(hog_feat)}")
            
            with col2:
                st.markdown("#### 📊 Feature Distribution")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.hist(hog_feat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_title('HOG Feature Values Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Feature Value', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
                st.markdown("**Statistics:**")
                st.write(f"- Mean: {np.mean(hog_feat):.4f}")
                st.write(f"- Std Dev: {np.std(hog_feat):.4f}")
                st.write(f"- Min: {np.min(hog_feat):.4f}")
                st.write(f"- Max: {np.max(hog_feat):.4f}")
        
        with tab2:
            st.markdown("### Local Binary Patterns (LBP)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 LBP Heatmap")
                radius, n_points = 2, 16
                lbp_image = local_binary_pattern(img_gray, n_points, radius, method='uniform')
                
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(lbp_image, cmap='hot')
                ax.set_title('LBP Texture Map', fontsize=14, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, label='LBP Value')
                st.pyplot(fig)
                
                st.info(f"**Total LBP Features**: {len(lbp_feat)}")
            
            with col2:
                st.markdown("#### 📊 LBP Histogram")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.bar(range(len(lbp_feat)), lbp_feat, color='coral', edgecolor='black')
                ax.set_title('LBP Feature Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Bin Index', fontweight='bold')
                ax.set_ylabel('Normalized Frequency', fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
                st.pyplot(fig)
                
                st.markdown("**Statistics:**")
                st.write(f"- Mean: {np.mean(lbp_feat):.4f}")
                st.write(f"- Std Dev: {np.std(lbp_feat):.4f}")
                st.write(f"- Entropy: {-np.sum(lbp_feat * np.log2(lbp_feat + 1e-10)):.4f}")
        
        with tab3:
            st.markdown("### Color Histogram (RGB Channels)")
            
            # Split histogram by channels
            hist_b = hist_feat[:32]
            hist_g = hist_feat[32:64]
            hist_r = hist_feat[64:]
            
            st.markdown("#### 🎨 RGB Channel Distribution")
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(32)
            ax.plot(x, hist_b, 'b-', label='Blue Channel', linewidth=2.5, marker='o', markersize=4)
            ax.plot(x, hist_g, 'g-', label='Green Channel', linewidth=2.5, marker='s', markersize=4)
            ax.plot(x, hist_r, 'r-', label='Red Channel', linewidth=2.5, marker='^', markersize=4)
            ax.set_title('Color Histogram (32 bins per channel)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Bin Index', fontweight='bold')
            ax.set_ylabel('Normalized Frequency', fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔵 Blue Channel**")
                st.write(f"- Mean: {np.mean(hist_b):.4f}")
                st.write(f"- Max: {np.max(hist_b):.4f}")
                st.write(f"- Dominant bin: {np.argmax(hist_b)}")
            
            with col2:
                st.markdown("**🟢 Green Channel**")
                st.write(f"- Mean: {np.mean(hist_g):.4f}")
                st.write(f"- Max: {np.max(hist_g):.4f}")
                st.write(f"- Dominant bin: {np.argmax(hist_g)}")
            
            with col3:
                st.markdown("**🔴 Red Channel**")
                st.write(f"- Mean: {np.mean(hist_r):.4f}")
                st.write(f"- Max: {np.max(hist_r):.4f}")
                st.write(f"- Dominant bin: {np.argmax(hist_r)}")
        
        with tab4:
            st.markdown("### 📊 Feature Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Feature Composition")
                
                feature_sizes = {
                    'HOG': len(hog_feat),
                    'LBP': len(lbp_feat),
                    'Color Histogram': len(hist_feat)
                }
                
                fig, ax = plt.subplots(figsize=(8, 8))
                colors_pie = ['#3498db', '#e74c3c', '#f39c12']
                wedges, texts, autotexts = ax.pie(
                    feature_sizes.values(), 
                    labels=feature_sizes.keys(),
                    autopct='%1.1f%%', 
                    colors=colors_pie, 
                    startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'}
                )
                ax.set_title('Feature Vector Composition', fontsize=14, fontweight='bold', pad=20)
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 📋 Feature Statistics")
                
                total_features = len(hog_feat) + len(lbp_feat) + len(hist_feat)
                combined_features = np.concatenate([hog_feat, lbp_feat, hist_feat])
                
                st.markdown(f"**Total Raw Features**: {total_features}")
                st.markdown(f"**After PCA**: ~200 features")
                st.markdown(f"**Dimensionality Reduction**: {(1 - 200/total_features)*100:.1f}%")
                
                st.markdown("---")
                
                st.markdown("**Combined Statistics:**")
                st.write(f"- Mean: {np.mean(combined_features):.4f}")
                st.write(f"- Std Dev: {np.std(combined_features):.4f}")
                st.write(f"- Min: {np.min(combined_features):.4f}")
                st.write(f"- Max: {np.max(combined_features):.4f}")
                
                st.markdown("---")
                st.info("You can save these extracted features or use them to visualize model decision boundaries in the Model Performance page.")

# ==================== PREDICTION PAGE ====================
elif page == "🎯 Prediction":
    st.markdown('<p class="sub-header">Upload CT Scan for Prediction</p>', unsafe_allow_html=True)
    st.markdown("Upload a single CT image (png/jpg/jpeg). Models must be loaded first (sidebar).")
    
    # Allow user to upload models if not loaded
    if not st.session_state.models_loaded:
        st.warning("Models not loaded. You can upload model files (models.pkl, scalers.pkl, pca.pkl, label_encoder.pkl) here.")
        uploaded_models = st.file_uploader("Upload model files (.pkl) — you can upload multiple", type=['pkl'], accept_multiple_files=True)
        if uploaded_models:
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            saved = []
            for up in uploaded_models:
                save_path = os.path.join(models_dir, up.name)
                with open(save_path, "wb") as f:
                    f.write(up.getbuffer())
                saved.append(up.name)
            if saved:
                st.success(f"Saved: {', '.join(saved)}. Click 'Load Pre-trained Models' in the sidebar.")
    
    uploaded_file = st.file_uploader("Upload CT image for prediction", type=['png','jpg','jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(preprocess_image(image)[0], cv2.COLOR_BGR2RGB), caption="Input (resized)", use_container_width=True)
        
        if not st.session_state.models_loaded:
            st.error("Models not loaded. Upload and load models from the sidebar first.")
        else:
            with st.spinner("Running prediction..."):
                try:
                    preds, feats = predict_image(image, st.session_state.models, st.session_state.scalers, st.session_state.pca, st.session_state.label_encoder)
                    st.success("Prediction complete.")
                    
                    # Display model-wise predictions
                    for model_name, res in preds.items():
                        card_class = "normal-card" if res['class'].lower() == 'normal' else ("bleeding-card" if 'bleed' in res['class'].lower() else "ischemia-card")
                        st.markdown(f'<div class="prediction-card {card_class}"><b>{model_name}</b> — Predicted: <b>{res["class"]}</b> ({res["confidence"]:.1f}%)</div>', unsafe_allow_html=True)
                        
                        # Expand probabilities
                        with st.expander(f"Probabilities — {model_name}"):
                            probs_df = pd.DataFrame(list(res['probabilities'].items()), columns=['Class','Probability'])
                            st.table(probs_df.style.format({'Probability': '{:.3f}'}))
                    
                    # Show extracted features summary
                    hog_feat, lbp_feat, hist_feat = feats
                    st.markdown("### Extracted Features Summary")
                    st.write(f"HOG length: {len(hog_feat)}, LBP length: {len(lbp_feat)}, Color hist length: {len(hist_feat)}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "📈 Model Performance":
    st.markdown('<p class="sub-header">Model Evaluation & Metrics</p>', unsafe_allow_html=True)
    
    # Try to load sample dataset for evaluation
    df = load_sample_dataset()
    if df is None:
        st.warning("Sample extracted_features.csv not found in ./sample_data/. Upload CSV named 'extracted_features.csv' there or via sidebar.")
        st.info("If you have true labels and features, place CSV in sample_data directory with 'label' column.")
    else:
        st.success("Sample dataset loaded.")
        st.dataframe(df.head(), use_container_width=True)
        
        if not st.session_state.models_loaded:
            st.warning("Load models to evaluate on sample dataset.")
        else:
            target_col = st.selectbox("Select target/label column", options=[c for c in df.columns], index=len(df.columns)-1)
            feature_cols = [c for c in df.columns if c != target_col]
            if st.button("Run evaluation on loaded sample dataset"):
                from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
                X = df[feature_cols].values
                y = df[target_col].values
                # simple evaluation using first model in dict
                model_name, model_obj = list(st.session_state.models.items())[0]
                try:
                    y_pred = model_obj.predict(X)
                    cr = classification_report(y, y_pred, output_dict=True)
                    cm = confusion_matrix(y, y_pred, labels=st.session_state.label_encoder.classes_)
                    st.markdown("#### Classification Report")
                    st.text(classification_report(y, y_pred))
                    
                    fig, ax = plt.subplots(figsize=(6,6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=st.session_state.label_encoder.classes_)
                    disp.plot(ax=ax, cmap='Blues', values_format='d')
                    ax.set_title(f'Confusion Matrix — {model_name}')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.markdown('<p class="sub-header">About This Project</p>', unsafe_allow_html=True)
    st.markdown("""
    - Project: AI Stroke Detection & Classification System  
    - Authors: Your Team / University Project  
    - Used features: HOG, LBP, Color Histogram, PCA, StandardScaler/MinMaxScaler  
    - Models included: SVM, KNN, Decision Tree, XGBoost
    
    Tips:
    - Ensure uploaded model files are compatible scikit-learn/XGBoost pickles.
    - For reproducibility keep the same preprocessing (scalers/pca/encoders) used during training.
    """)
    st.markdown("---")
    st.markdown("### Contact / Notes")
    st.write("This demo is for educational purposes. Validate thoroughly before any clinical use.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888;'>© Project KA — AI Stroke Detection — Not for clinical use</div>", unsafe_allow_html=True)