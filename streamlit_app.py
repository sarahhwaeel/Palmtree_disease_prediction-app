import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import os

# Function to download and load model from URL or local file
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/sarahhwaeel/Streamlit-prediction-app/releases/download/%23v1.0.0/palmtree_disease_model.h5"
    model_path = "palmtree_disease_model.h5"
    if not os.path.exists(model_path):
        st.info("Downloading model... This might take a moment.")
        urllib.request.urlretrieve(model_url, model_path)
    return tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert image to grayscale
    img_gray = image.convert('L')
    
    # Resize the image (example resizing to 256x256)
    img_resized = img_gray.resize((256, 256))
    
    # Normalize pixel values
    img_array = np.array(img_resized) / 255.0
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict disease class and suggest pesticide
def predict_disease_and_pesticide(model, img_array):
    # Make prediction
    predictions = model.predict(img_array)
    class_labels = ['brown spots', 'healthy', 'white scale']
    predicted_class = class_labels[np.argmax(predictions)]
    
    # Determine pesticide suggestion based on predicted class
    if predicted_class == 'brown spots':
        pesticide_info = "Use Fungicidal sprays containing copper."
    elif predicted_class == 'healthy':
        pesticide_info = "No pesticide used."
    elif predicted_class == 'white scale':
        pesticide_info = "Use Chemical insecticides as buprofezin or pyriproxyfen."
    else:
        pesticide_info = "No specific recommendation."
    
    return predicted_class, pesticide_info

# Main app
def main():
    st.title("Palm Tree Disease Prediction and Pesticide Suggestion")

    # Load the pre-trained model
    model = load_model()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)

            # Preprocess the image
            img_array = preprocess_image(img)

            # Make prediction and get pesticide suggestion
            if model is not None:
                predicted_class, pesticide_info = predict_disease_and_pesticide(model, img_array)
                st.markdown(f"**<h3 style='font-size:24px'>Predicted Class: {predicted_class}</h3>**", unsafe_allow_html=True)
                st.markdown(f"**<h3 style='font-size:24px'>Pesticide suggested: {pesticide_info}</h3>**", unsafe_allow_html=True)
            else:
                st.warning("Model loading failed. Please check your model.")

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
