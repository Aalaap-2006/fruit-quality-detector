import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import time

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Fruit Quality Detector", layout="centered")

# -------------------------
# CSS Styling
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.95)),
    url("https://images.unsplash.com/photo-1615486363973-7f6b3e5f3e58");
    background-size: cover;
    color: white;
}

h1 {
    text-align: center;
    color: #00FFAA;
}

/* Glass card */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
}

/* Result boxes */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.fresh {
    background: rgba(0,255,0,0.2);
    color: #00FF00;
}

.rotten {
    background: rgba(255,0,0,0.2);
    color: #FF4C4C;
}

/* Button styling */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #00FFAA;
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title + Description
# -------------------------
st.markdown("<h1>🍎 Fruit Quality Detector</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 16px; color: #CCCCCC; margin-bottom: 20px;'>

This application uses a deep learning model to detect whether a fruit is <b>fresh</b> or <b>rotten</b>.

<br><br>

<b>How it works:</b><br>
1️⃣ Choose input method (Upload / Camera)<br>
2️⃣ Provide a fruit image<br>
3️⃣ Click Predict<br>
4️⃣ Model analyzes features<br>
5️⃣ Get freshness result with confidence

</div>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fruit_mobilenet_final.h5")

model = load_model()

# Load class names
with open("model/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# -------------------------
# Input Section
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### 📸 Choose Input Method")

use_camera = st.toggle("📷 Turn ON Camera")

uploaded_file = None
camera_image = None

if use_camera:
    camera_image = st.camera_input("Take a Picture")
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Select Image
# -------------------------
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif camera_image is not None:
    image = Image.open(camera_image)

# -------------------------
# Prediction Section
# -------------------------
if image is not None:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.image(image, caption="Selected Image", use_column_width=True)

    if st.button("🔍 Predict"):

        # Loading animation
        with st.spinner("Analyzing fruit..."):
            time.sleep(1.5)

        img = image.convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        # 🔥 Extract only quality
        if "fresh" in predicted_class:
            quality = "Fresh"
        else:
            quality = "Rotten"

        st.markdown("---")

        # Result display
        if quality == "Fresh":
            st.markdown(
                f"<div class='result-box fresh'>🟢 {quality}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box rotten'>🔴 {quality}</div>",
                unsafe_allow_html=True
            )

        # Confidence bar
        st.markdown("### Confidence Level")
        st.progress(int(confidence))

        # Details
        st.markdown("### 📊 Details")
        st.write(f"Quality: **{quality}**")
        st.write(f"Confidence Score: **{confidence:.2f}%**")

    st.markdown("</div>", unsafe_allow_html=True)