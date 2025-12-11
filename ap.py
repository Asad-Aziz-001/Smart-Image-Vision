import streamlit as st
from PIL import Image
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Image Vision",
    page_icon="🤖📷",
    layout="centered",
)

# ---------------------------------------------------------
# ADVANCED MODERN GRADIENT UI (2025 Level)
# ---------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #ffffff;
}

/* Header */
h1 {
    background: linear-gradient(to right, #FF8CFF, #60AFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

/* Sidebar */
.stSidebar {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Sidebar Text */
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p {
    color: #e8e8e8 !important;
}

/* Cards */
.stFileUploader > div {
    background: rgba(255,255,255,0.10);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(12px);
}

/* Image */
.stImage img {
    border-radius: 12px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    color: white;
    border-radius: 10px;
    padding: 12px 26px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #8aa8ff, #c58aff);
    transform: scale(1.04);
}

/* Success box */
.stSuccess {
    background: rgba(76, 175, 80, 0.15);
    border-left: 5px solid #4CAF50;
}

/* Result Text */
.result-box {
    background: rgba(255,255,255,0.1);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Footer */
footer, .stCaption {
    color: #dcdcdc !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("✨ About This App")
    st.write("""
    This AI analyzes your uploaded images using deep learning.
    
    It generates:
    - 📝 Description  
    - 🎨 Dominant Colors  

    Built with **Transformers**, **Streamlit**, and **ML models**.
    """)

# ---------------------------------------------------------
# LOAD BLIP MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ---------------------------------------------------------
# COLOR DETECTION
# ---------------------------------------------------------
def detect_dominant_color(image, k=3):
    img = np.array(image)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img)

    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    dominant = colors[counts.argmax()]

    return tuple(map(int, dominant))

def rgb_to_name(rgb):
    r, g, b = rgb

    if r > 200 and g > 200 and b > 200: return "white"
    if r < 50 and g < 50 and b < 50: return "black"
    if r > 150 and g < 100 and b < 100: return "red"
    if r < 100 and g > 150 and b < 100: return "green"
    if r < 100 and g < 100 and b > 150: return "blue"
    if r > 150 and g > 150 and b < 100: return "yellow"
    if r > 150 and g < 100 and b > 150: return "purple"
    if r < 100 and g > 150 and b > 150: return "cyan"
    return "unknown"

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
st.title("🤖 Smart Image Vision")
st.write("Upload an image and let AI analyze it with modern deep-vision intelligence.")

uploaded = st.file_uploader(
    "Upload your image", type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("⏳ Analyzing image with AI..."):
            # Caption
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Color
            dom_rgb = detect_dominant_color(image)
            dom_name = rgb_to_name(dom_rgb)

        st.success("✔ Analysis Complete")

        st.markdown("### 📝 Description")
        st.markdown(f"<div class='result-box'>{caption}</div>", unsafe_allow_html=True)

        st.markdown("### 🎨 Dominant Color")
        st.markdown(
            f"<div class='result-box'><b>{dom_name}</b> — RGB {dom_rgb}</div>",
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by **ASAD AZIZ** | Modern UI Powered by Streamlit + Custom CSS")

