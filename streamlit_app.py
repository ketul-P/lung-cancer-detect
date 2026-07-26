import streamlit as st
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions



# Function to clear uploaded image
def clear_image():
    st.uploaded_image = None

# Load the model
def load_model():
    return tf.keras.models.load_model('best_model1.keras')

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

with st.bottom:
    st.caption("Built by Ketul Patel · Revised 2026")

# Load model
model = load_model()

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

st.markdown("This project uses a Convolutional Neural Network trained on labeled lung scan images to classify scans for signs of cancer.\n")

st.markdown("""
### How to use the demo tool
- Upload a chest CT scan image or select one of the provided sample images to get started.
- Click Diagnose to run the AI model, which will analyze the image and predict whether it is Normal or NSCLC (Non-Small Cell Lung Cancer).
- The prediction will be displayed within a few seconds!""")

st.subheader("Try a Sample CT Scan")

samples = [
    ("sample_images/adenocarcinoma/ct-scan_chest_" + str(random.randint(4, 6)) + ".png", "Adenocarcinoma"),
    ("sample_images/large_cell/ct-scan_chest_" + str(random.randint(7, 9)) + ".png", "Large Cell"),
    ("sample_images/normal/ct-scan_chest_"+ str(random.randint(1,3)) +".png", "Normal"),
    ("sample_images/squamous/ct-scan_chest_" + str(random.randint(13, 15)) + ".png", "Squamous Cell")
]

cols = st.columns(2)

for i, (path, label) in enumerate(samples):

    with cols[i % 2]:

        with st.container(border=True, height=220,  horizontal_alignment="center", vertical_alignment="center"):

            st.image(
                path,
                width=150
            )

            if st.button(
                f"Use {label}",
                key=f"sample_{i}",
                use_container_width=True
            ):

                st.session_state.selected_image = path

st.subheader("Upload your own CT Scan")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width = 300)

image_to_use = uploaded_file

if image_to_use is None:
    image_to_use = st.session_state.selected_image

if image_to_use is not None:
    st.image(
        image_to_use,
        caption="Selected Image",
        width=300
    )
# Create a horizontal layout for buttons
if st.button("Diagnose", help="Perform Diagnosis", width=120):
    diagnose_image(image_to_use, model)
