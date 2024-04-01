import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

# Function to make predictions
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(image):
    # Preprocess the image (resize, normalize, etc.)
    processed_image = preprocess_image(image)
    # Make predictions using the loaded model
    predictions = model.predict(processed_image)
    return predictions
# Function to clear uploaded image
def clear_image():
    st.session_state.uploaded_image = None

# Function to diagnose uploaded image
def diagnose_image():
    # Your diagnosis logic goes here
    st.session_state.diagnosis_result = "Diagnosis: Placeholder result"

# Streamlit app
st.title("Medical Image Diagnosis")

# Initialize session state variables
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = ""

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="upload_button")

# Display uploaded image
if uploaded_file is not None:
    st.session_state.uploaded_image = uploaded_file

if st.session_state.uploaded_image is not None:
    st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)


# Create a horizontal layout for buttons
col1, col2 = st.columns([1, 1])

# Clear button
with col1:
    if st.button("Clear", key="clear_button", help="Clear Image", use_container_width=True):
        clear_image()

# Diagnose button
with col2:
    if st.button("Diagnose", key="diagnose_button", help="Perform Diagnosis", use_container_width=True):
        diagnose_image()

# Display diagnosis result
st.write(st.session_state.diagnosis_result)
