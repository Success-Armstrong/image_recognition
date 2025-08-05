
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fake_image_cnn.h5")

model = load_model()

def preprocess(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("AI-Based Image Manipulation Detector")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(uploaded, caption="Uploaded Image", use_container_width=True)


    st.write("Analyzing...")
    input_image = preprocess(image)
    prediction = model.predict(input_image)[0][0]

    label = "Manipulated" if prediction > 0.5 else "Authentic"
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {prediction*100:.2f}%")


from pyngrok import ngrok

