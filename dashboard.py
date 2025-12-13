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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog, local_binary_pattern
from PIL import Image
import io
import zipfile
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="AI Stroke Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Updated for High Contrast
st.markdown("""
<style>
    /* Header Utama */
    .main-header {
        font-size: 2.5rem;
        color: #4da6ff; /* Biru lebih terang agar kontras di dark mode */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ffa726; /* Orange terang */
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #262730; /* Warna gelap standar Streamlit */
        border: 1px solid #4da6ff; /* Border biru */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Paksa teks di dalam metric card jadi putih (opsional, defaultnya sudah putih) */
    .metric-card div {
        color: white !important;
    }

    .prediction-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000000 !important; 
        font-weight: 500;
    }
    
    .normal-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724 !important; /* Hijau tua gelap */
    }
    .bleeding-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24 !important; /* Merah tua gelap */
    }
    .ischemia-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404 !important; /* Kuning tua gelap */
    }
    
    .loading-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        color: #0c5460 !important; /* Biru tua gelap */
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive Links (Ganti dengan link Anda yang sudah di-share publicly)
GDRIVE_LINKS = {
    # Model files
    'svm': 'https://drive.google.com/uc?id=1537b6LEIgRZFBOfULIDyYvSntBS07MXe',
    'knn': 'https://drive.google.com/uc?id=1RPOaAAJFyvT0qVfWpNMxZUcoOlem8iUW',
    'dt':  'https://drive.google.com/uc?id=19digDS8Ihn4n6oVccqgZgypRV_ckD7oj',
    'xgb': 'https://drive.google.com/uc?id=1HjGTtRtvnALyvGlXyPOmJ14R5t08oWtn',
    'scaler_hog': 'https://drive.google.com/uc?id=1qNCcygPQpTs4K_KewD5yXRb4rSLHA_Ny',
    'scaler_lbp': 'https://drive.google.com/uc?id=1H-JHD7_SDozlt0hSjQTF7r5Kppcj9Uy3',
    'scaler_hist': 'https://drive.google.com/uc?id=1NePlUaoFgxj49JL6_lZ3mDCR_TVckTQ2',
    'pca': 'https://drive.google.com/uc?id=15Fjd3jKVhJm6HILOCJ8N7_57t01LPnle',
    'label_encoder': 'https://drive.google.com/uc?id=14b3d_sJPytdQStrnaxHW4I8l6EAvFy3j',
    'extracted_features': 'https://drive.google.com/uc?id=1r_gItSWbcYf94_-gVXgVksCzoQYp5-dg',
    'sample_images_zip': 'https://drive.google.com/uc?id=1ee21Y0VDjhbjfMXkXr2GDnFwcOYZ54Yn',
    'full_dataset_zip': 'https://drive.google.com/uc?id=1ee21Y0VDjhbjfMXkXr2GDnFwcOYZ54Yn',
}

# Directory structure
DIRS = {
    'models': 'models',
    'data': 'data',
    'sample_images': 'data/sample_images',
    'dataset': 'data/dataset'
}

# Create directories
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

def initialize_session_state():
    defaults = {
        'models_loaded': False,
        'dataset_loaded': False,
        'models': {},
        'scalers': {},
        'pca': None,
        'label_encoder': None,
        'class_names': [],
        'sample_df': None,
        'sample_images': {},
        'initialization_done': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

@st.cache_data(show_spinner=False)
def download_file_from_gdrive(url, output_path, description="file"):
    try:
        if not url or 'YOUR_' in url:
            return False, f"{description} URL not configured"
        
        if os.path.exists(output_path):
            return True, f"{description} already exists"
        
        # Create parent directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download
        gdown.download(url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            return True, f"{description} downloaded successfully"
        else:
            return False, f"Failed to download {description}"
            
    except Exception as e:
        return False, f"Error downloading {description}: {str(e)}"

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True, f"Extracted to {extract_to}"
    except Exception as e:
        return False, f"Extraction error: {str(e)}"

@st.cache_resource(show_spinner=False)
def auto_load_models():
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.markdown('<div class="loading-box">🔄 <b>Initializing AI System...</b></div>', unsafe_allow_html=True)
    
    models_dir = DIRS['models']
    
    # Model file mappings
    model_files = {
        'svm': 'svm_model.pkl',
        'knn': 'knn_model.pkl',
        'dt':  'dt_model.pkl',
        'xgb': 'xgboost_model.pkl',
        'scaler_hog': 'scaler_hog.pkl',
        'scaler_lbp':  'scaler_lbp.pkl',
        'scaler_hist': 'scaler_hist.pkl',
        'pca': 'pca.pkl',
        'label_encoder': 'label_encoder.pkl'
    }
    
    # Download all model files
    total_files = len(model_files)
    progress_bar = st.progress(0)
    
    for idx, (key, filename) in enumerate(model_files.items()):
        file_path = os.path.join(models_dir, filename)
        
        status_placeholder.info(f"📥 Downloading {filename}... ({idx+1}/{total_files})")
        
        if key in GDRIVE_LINKS:
            success, msg = download_file_from_gdrive(
                GDRIVE_LINKS[key], 
                file_path, 
                filename
            )
            
            if not success and not os.path.exists(file_path):
                st.warning(f"⚠️ {msg}")
        
        progress_bar.progress((idx + 1) / total_files)
    
    # Load models
    models = {}
    scalers = {}
    pca = None
    label_encoder = None
    
    status_placeholder.info("🔧 Loading models into memory...")
    
    # Load ML models
    for model_name in ['svm', 'knn', 'dt', 'xgb']:
        model_path = os.path.join(models_dir, model_files[model_name])
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Could not load {model_name}: {e}")
    
    # Load scalers
    if not scalers:
        for feat_name in ['hog', 'lbp', 'hist']:
            scaler_path = os.path.join(models_dir, f'scaler_{feat_name}.pkl')
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scalers[feat_name] = pickle.load(f)
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {feat_name} scaler: {e}")

    if 'hog' not in scalers:
        st.warning("⚠️ HOG scaler missing, creating fitted dummy (identity transform)")
        scalers['hog'] = StandardScaler()
        scalers['hog'].mean_ = np.zeros(8100)
        scalers['hog'].scale_ = np.ones(8100)
        scalers['hog'].var_ = np.ones(8100)
        scalers['hog'].n_features_in_ = 8100
        scalers['hog'].n_samples_seen_ = 1000

    if 'lbp' not in scalers:
        st.warning("⚠️ LBP scaler missing, creating fitted dummy (identity transform)")
        scalers['lbp'] = StandardScaler()
        scalers['lbp'].mean_ = np.zeros(18)
        scalers['lbp'].scale_ = np.ones(18)
        scalers['lbp'].var_ = np.ones(18)
        scalers['lbp'].n_features_in_ = 18
        scalers['lbp'].n_samples_seen_ = 1000

    if 'hist' not in scalers:
        st.warning("⚠️ Hist scaler missing, creating fitted dummy (identity transform)")
        scalers['hist'] = MinMaxScaler()
        scalers['hist'].min_ = np.zeros(96)
        scalers['hist'].scale_ = np.ones(96)
        scalers['hist'].data_min_ = np.zeros(96)
        scalers['hist'].data_max_ = np.ones(96)
        scalers['hist'].data_range_ = np.ones(96)
        scalers['hist'].n_features_in_ = 96
        scalers['hist'].n_samples_seen_ = 1000
    
    # Load PCA
    pca_path = os.path.join(models_dir, model_files['pca'])
    if os.path.exists(pca_path):
        try:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to load PCA: {e}")
    
    # Load label encoder
    le_path = os.path.join(models_dir, model_files['label_encoder'])
    if os.path.exists(le_path):
        try:
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to load label encoder: {e}")
    
    progress_placeholder.empty()
    status_placeholder.empty()
    
    if not models:
        return None, None, None, None, False
    
    return models, scalers, pca, label_encoder, True

@st.cache_data(show_spinner=False)
def auto_load_dataset():
    status_placeholder = st.empty()
    
    # Load extracted features CSV
    features_path = os.path.join(DIRS['data'], 'extracted_features.parquet')
    
    if not os.path.exists(features_path) and 'extracted_features' in GDRIVE_LINKS:
        status_placeholder.info("📥 Downloading extracted features dataset...")
        success, msg = download_file_from_gdrive(
            GDRIVE_LINKS['extracted_features'],
            features_path,
            "extracted_features.parquet"
        )
    
    # Load CSV
    df = None
    if os.path.exists(features_path):
        try:
            df = pd.read_parquet(features_path)
            status_placeholder.success(f"✅ Dataset loaded: {len(df)} samples")
        except Exception as e:
            status_placeholder.warning(f"⚠️ Could not load CSV: {e}")
    
    # Download and extract sample images
    sample_zip_path = os.path.join(DIRS['data'], 'sample_images.zip')
    
    if not os.listdir(DIRS['sample_images']) and 'sample_images_zip' in GDRIVE_LINKS:
        status_placeholder.info("📥 Downloading sample images...")
        success, msg = download_file_from_gdrive(
            GDRIVE_LINKS['sample_images_zip'],
            sample_zip_path,
            "sample_images.zip"
        )
        
        if success and os.path.exists(sample_zip_path):
            status_placeholder.info("📦 Extracting sample images...")
            extract_zip(sample_zip_path, DIRS['sample_images'])

            try:
                os.remove(sample_zip_path)
                status_placeholder.info("🗑️ Removed zip file after extraction.")
            except:
                pass
    status_placeholder.empty()
    
    # Load sample images into memory
    sample_dir = Path(DIRS['sample_images'])
    if sample_dir.exists():
        for img_file in sample_dir.rglob('*.png'):
            try:
                sample_images[img_file.stem] = Image.open(img_file)
            except:
                pass
        
        for img_file in sample_dir.rglob('*.jpg'):
            try:
                sample_images[img_file.stem] = Image.open(img_file)
            except:
                pass
    
    status_placeholder.empty()
    
    return df, DIRS['sample_images']

# Initialize dynamic dataset stats di awal file, setelah auto_load_dataset()
def get_dataset_stats():
    df = st.session_state.sample_df 
    
    if df is not None:
        # Auto-detect label column
        label_col = None
        for col in ['label', 'class', 'target', 'y']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            # Assume last column is label
            label_col = df.columns[-1]
        
        return {
            'total_images': len(df),
            'classes': df[label_col].nunique(),
            'features': len([c for c in df.columns if c != label_col])
        }
    
    return {
        'total_images': 0,
        'classes': 3,
        'features': 0
    }

if not st.session_state.initialization_done:
    with st.spinner("🚀 Initializing AI Stroke Detection System..."):
        # Auto-load models
        models, scalers, pca, label_encoder, success = auto_load_models()
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        class_names = []

        if label_encoder is not None:
            if hasattr(label_encoder, "classes_"):
                class_names = list(label_encoder.classes_)

            elif isinstance(label_encoder, (list, tuple, np.ndarray)):
                le_new = LabelEncoder()
                le_new.fit(label_encoder)
                label_encoder = le_new
                class_names = list(le_new.classes_)

            else:
                class_names = []
        else:
            class_names = []

        if success:
            st.session_state.models = models
            st.session_state.scalers = scalers
            st.session_state.pca = pca
            manual_class_names = ["Bleeding", "Ischemia", "Normal"]

            from sklearn.preprocessing import LabelEncoder
            le_fixed = LabelEncoder()
            le_fixed.fit(manual_class_names)
            label_encoder = le_fixed
            class_names = le_fixed.classes_.tolist()
            st.session_state.label_encoder = le_fixed
            st.session_state.class_names = class_names
            st.session_state.models_loaded = True

        # Auto-load dataset
        df, sample_images = auto_load_dataset()
        st.session_state.sample_df = df
        st.session_state.sample_images = sample_images
        st.session_state.dataset_loaded = (df is not None)

        st.session_state.initialization_done = True
        st.rerun()


def extract_features(img_color, img_gray):
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
    img_resized = cv2.resize(image, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_resized, img_gray

def predict_image(image, models, scalers, pca, label_encoder):
    # Preprocess image
    img_resized, img_gray = preprocess_image(image)
    
    # Extract features
    hog_feat, lbp_feat, hist_feat = extract_features(img_resized, img_gray)
    
    # Normalize features separately
    hog_norm = scalers['hog'].transform([hog_feat])
    lbp_norm = scalers['lbp'].transform([lbp_feat])
    hist_norm = scalers['hist'].transform([hist_feat])
    
    # Apply PCA
    if pca is not None:
        hog_pca = pca.transform(hog_norm)
        X_pred = np.hstack([hog_pca, lbp_norm, hist_norm])
    else:
        X_pred = np.hstack([hog_norm, lbp_norm, hist_norm])
    
    # Make predictions
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_pred)[0]

        # FIX: pastikan string
        try:
            pred_class = label_encoder.inverse_transform([pred])[0]
        except:
            pred_class = pred

        pred_class = str(pred_class)  # <--- solusi utama

        proba = model.predict_proba(X_pred)[0]
        confidence = float(np.max(proba) * 100)

        # FIX: pastikan key probability juga STR
        class_labels = [str(c) for c in label_encoder.classes_] if hasattr(label_encoder, "classes_") else []
        
        predictions[name] = {
            'class': pred_class,
            'confidence': confidence,
            'probabilities': dict(zip(class_labels, proba))
        }
    
    return predictions, (hog_feat, lbp_feat, hist_feat)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
    st.title("🧠 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📊 Dataset Overview", "🔬 Feature Extraction", 
        "🎯 Prediction", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    
    # System status indicators
    if st.session_state.models_loaded:
        st.success(f"✅ Models: {len(st.session_state.models)} loaded")
    else:
        st.error("❌ Models: Not loaded")
    
    if st.session_state.dataset_loaded:
        st.success(f"✅ Data Features: Loaded")
    
    if st.session_state.sample_images and os.path.exists(st.session_state.sample_images):
        st.success(f"✅ Images: Ready on disk")
    else:
        st.info("ℹ️ Images: Downloading...")
    
    # Refresh button
    if st.button("🔄 Refresh System", type="secondary"):
        st.session_state.initialization_done = False
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("- [📁 Google Drive Dataset](https://drive.google.com/drive/folders/1arvBtDxdOE8-7caIXArXVTxOoS1vSUgd)")
    st.markdown("- [💻 GitHub Repository](https://github.com/JullMol/Artifical-Intelligence-Project)")
    
    st.markdown("---")
    st.info("💡 **Auto-Load**: Models and data load automatically on startup")

# Header
st.markdown('<p class="main-header">🧠 AI Stroke Detection & Classification System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Turkish Ministry of Health Open Dataset Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    stats = get_dataset_stats()
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
        
        st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Dataset Size</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{stats['total_images']} Images</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Classes</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{stats['classes']} Types</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Best Accuracy: SVM Tuned</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">98%</div>
                </div>
                """, unsafe_allow_html=True)
        
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
    stats = get_dataset_stats()
    col1, col2, col3 = st.columns(3)
    
    with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.8;">Total Images</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{stats['total_images']}</div>
            </div>
            """, unsafe_allow_html=True)
        
    with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.8;">Image Size</div>
                <div style="font-size: 1.8rem; font-weight: bold;">128×128 px</div>
            </div>
            """, unsafe_allow_html=True)
        
    with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.8;">Classes</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{stats['classes']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Class Distribution
    st.markdown("### 📊 Class Distribution")
    
    # Use real dataset if available, otherwise fallback to hardcoded sample
    if st.session_state.sample_df is not None:
        df_dist = st.session_state.sample_df.copy()
        # try common label column names
        if 'label' in df_dist.columns:
            label_col = 'label'
        elif 'class' in df_dist.columns:
            label_col = 'class'
        else:
            # assume last column is label if none named explicitly
            label_col = df_dist.columns[-1]
        
        counts = df_dist[label_col].value_counts().sort_index()
        
        # if numeric labels and label_encoder available, map to class names
        try:
            if pd.api.types.is_integer_dtype(df_dist[label_col]) or pd.api.types.is_numeric_dtype(df_dist[label_col]):
                le = st.session_state.label_encoder if 'label_encoder' in st.session_state else None
                if le is not None:
                    mapped_index = le.inverse_transform(counts.index.astype(int))
                    counts.index = mapped_index
        except Exception:
            # ignore mapping errors and keep raw index
            pass
        
        df_classes = pd.DataFrame({
            'Class': counts.index.astype(str),
            'Count': counts.values
        })
        total = df_classes['Count'].sum()
        df_classes['Percentage'] = (df_classes['Count'] / total * 100).round(1)
    else:
        st.warning("Sample dataset belum tersedia — menggunakan contoh statis.")
        df_classes = pd.DataFrame({
            'Class': ['Bleeding', 'Ischemia', 'Normal'],
            'Count': [750, 800, 800],
            'Percentage': [31.9, 34.0, 34.0]
        })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#ff6b6b', '#feca57', '#48dbfb'][:len(df_classes)]
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
        explode = tuple([0.03]*len(df_classes))
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
    image_dir_path = st.session_state.sample_images

    all_image_files = []
    if isinstance(image_dir_path, str) and os.path.exists(image_dir_path):
        p = Path(image_dir_path)
        all_image_files = list(p.rglob('*.jpg')) + list(p.rglob('*.png')) + list(p.rglob('*.jpeg'))

    if all_image_files:
        num_display = 9
        st.write(f"Total images available on disk: {len(all_image_files)}")

        cols = st.columns(3)
        for idx, img_path in enumerate(all_image_files[:num_display]):
            with cols[idx % 3]:
                try:
                    img_obj = Image.open(img_path)
                    st.image(img_obj, caption=img_path.stem, use_container_width=True)
                    img_obj.close()
                except Exception as e:
                    st.error(f"Error loading image {img_path.name}: {e}")
    else:
        st.info("ℹ️ Images are not downloaded yet or folder is empty.")
    
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