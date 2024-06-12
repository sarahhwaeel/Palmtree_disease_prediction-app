import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import os

# Function to download and load model from URL or local file
@st.cache_resource
def load_model():
    model_url = "https://github.com/sarahhwaeel/Streamlit-prediction-app/releases/download/%23v1.0.0/palmtree_disease_model.h5"
    model_path = "palmtree_disease_model.h5"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Check the image format
    print("Image format:", image.format)
    
    # Convert to RGB if the image is in grayscale mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Main app
def main():
    st.title("Palm Tree Disease Prediction Application")

    # Load model
    model = load_model()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", width=300)

            # Preprocess the image
            img_array = preprocess_image(img)

            # Make prediction
            predictions = model.predict(img_array)
            class_labels = ['brown spots', 'healthy', 'white scale']
            predicted_class = class_labels[np.argmax(predictions)]

            # Display prediction
            st.markdown(f"**<h3 style='font-size:24px'>Predicted Class: {predicted_class}</h3>**", unsafe_allow_html=True)

            # Display pesticide suggestion
            if predicted_class == 'brown spots':
                pesticide_info = "Use Fungicidal sprays containing copper."
            elif predicted_class == 'healthy':
                pesticide_info = "No pesticide used."
            elif predicted_class == 'white scale':
                pesticide_info = "Use Chemical insecticides as buprofezin or pyriproxyfen."

            st.markdown(f"**<h3 style='font-size:24px'>Pesticide suggested: {pesticide_info}</h3>**", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()





