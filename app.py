import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import cv2


MODEL_PATH = "melanoma_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1XyRpIGwXgfcyA_ERsqmlGdMIqUeig3dH"



st.set_page_config(
    page_title="Melanoma Detection System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI-Powered Melanoma Detection")
st.markdown(
    "Upload a skin lesion image to predict whether it is **Benign** or **Malignant**."
)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()



def is_skin_image(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio > 0.03



st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.05
)

st.sidebar.markdown(
"""
Lower threshold → Higher cancer detection (higher recall)  
Higher threshold → Fewer false alarms (higher precision)
"""
)



uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if not is_skin_image(image):
            st.warning("⚠️ The uploaded image does not appear to contain skin. Please upload a skin lesion image.")
        else:
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("Analyzing image..."):
                prediction = float(model.predict(img_array)[0][0])

            st.subheader("🔍 Prediction Result")

            if prediction > threshold:
                confidence = prediction * 100
                st.error(
                    f"⚠️ **Malignant Detected**\n\n"
                    f"Confidence: {confidence:.2f}%"
                )
            else:
                confidence = (1 - prediction) * 100
                st.success(
                    f"✅ **Benign Detected**\n\n"
                    f"Confidence: {confidence:.2f}%"
                )

            st.markdown("---")
            st.write("### 📊 Probability Breakdown")
            st.write(f"Malignant Probability: {prediction*100:.2f}%")
            st.write(f"Benign Probability: {(1-prediction)*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")



st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This AI tool is for educational purposes only "
    "and is not a substitute for professional medical diagnosis."
)