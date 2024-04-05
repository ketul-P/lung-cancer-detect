import streamlit as st
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import tensorflow

# Function to clear uploaded image
def clear_image():
    st.uploaded_image = None

# Load the model
def load_model():
    return tensorflow.keras.models.load_model('best_model1.keras')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array = preprocess_input(img_array)  # Preprocess the image
    return img_array

# Function to diagnose uploaded image
def diagnose_image(uploaded_file, model):
    # Display uploaded image and perform diagnosis
    if uploaded_file is not None:
        # Preprocess the image
        processed_image = preprocess_image(image.load_img(uploaded_file, target_size=(224, 224)))

        # Make predictions
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        
        if predicted_class == 0:
            result = "Adenocarcinoma"
        elif predicted_class == 1:
            result = "Large Cell Carcinoma"
        elif predicted_class == 3: 
            result = "Squamous Cell Carcinoma"
        else:
            result = "Normal"
            # Display diagnosis result
            st.write("Diagnosis:", ':green[' + result + ']')
            return

        # Display diagnosis result
        st.write("Diagnosis", ':orange[' + result + ']')
        return

# Streamlit app
st.title("NSCLC Diagnosis")

# Load model
model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
  
# Create a horizontal layout for buttons
if st.button("Diagnose", help="Perform Diagnosis", use_container_width=True):
    diagnose_image(uploaded_file, model)
