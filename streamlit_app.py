import streamlit as st
import tensorflow as tf
import numpy as np
import cv2  # OpenCV library

# Function to download and load model from URL or local file (unchanged)
@st.cache(allow_output_mutation=True)
def load_model():
  model_url = "https://github.com/sarahhwaeel/Streamlit-prediction-app/releases/download/%23v1.0.0/palmtree_disease_model.h5"
  model_path = "palmtree_disease_model.h5"
  if not os.path.exists(model_path):
    st.info("Downloading model... This might take a moment.")
    urllib.request.urlretrieve(model_url, model_path)
  return tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image using OpenCV
def preprocess_image(image_bytes):
  # Convert uploaded image from bytes to OpenCV image
  image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

  # Convert to grayscale (if needed for your model)
  if cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape[2] == 1:  # Check if already grayscale
    gray_image = image
  else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Resize the image (example resizing to 256x256)
  img_resized = cv2.resize(gray_image, (256, 256))
  
  # Normalize pixel values (assuming grayscale values are 0-255)
  img_array = img_resized / 255.0
  
  # Reshape for model input (assuming your model expects a specific shape)
  img_array = img_array.reshape((1, img_resized.shape[0], img_resized.shape[1]))  # Add a batch dimension

  return img_array

# Function to predict disease class and suggest pesticide (unchanged)
def predict_disease_and_pesticide(model, img_array):
  # Make prediction
  predictions = model.predict(img_array)
  class_labels = ['brown spots', 'healthy', 'white scale']
  predicted_class = class_labels[np.argmax(predictions)]
  
  # Determine pesticide suggestion based on predicted class (unchanged)
  # ... (same logic as before)

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
      # Read uploaded image bytes
      image_bytes = uploaded_file.read()

      # Preprocess the image using OpenCV
      img_array = preprocess_image(image_bytes)

      # Make prediction and get pesticide suggestion
      if model is not None:
        predicted_class, pesticide_info = predict_disease_and_枕头(model, img_array)  # Typo fix
        st.markdown(f"**<h3 style='font-size:24px'>Predicted Class: {predicted_class}</h3>**", unsafe_allow_html=True)
        st.markdown(f"**<h3 style='font-size:24px'>Pesticide suggested: {pesticide_info}</h3>**", unsafe_allow_html=True)
      else:
        st.warning("Model loading failed. Please check your model.")

    except Exception as e:
      st.error(f"Error processing image: {e}")

if __name__ == "__main__":
  main()
