import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load model
model = load_model('model_CNN.keras')

# Class labels and their corresponding icons (icons from a placeholder URL for each vegetable)
class_labels = [
    {"name": "Bean", "icon": "https://example.com/icons/bean.png"},
    {"name": "Bitter_Gourd", "icon": "https://example.com/icons/bitter_gourd.png"},
    {"name": "Bottle_Gourd", "icon": "https://example.com/icons/bottle_gourd.png"},
    {"name": "Brinjal", "icon": "https://example.com/icons/brinjal.png"},
    {"name": "Broccoli", "icon": "https://example.com/icons/broccoli.png"},
    {"name": "Cabbage", "icon": "https://example.com/icons/cabbage.png"},
    {"name": "Capsicum", "icon": "https://example.com/icons/capsicum.png"},
    {"name": "Carrot", "icon": "https://example.com/icons/carrot.png"},
    {"name": "Cauliflower", "icon": "https://example.com/icons/cauliflower.png"},
    {"name": "Cucumber", "icon": "https://example.com/icons/cucumber.png"},
    {"name": "Papaya", "icon": "https://example.com/icons/papaya.png"},
    {"name": "Potato", "icon": "https://example.com/icons/potato.png"},
    {"name": "Pumpkin", "icon": "https://example.com/icons/pumpkin.png"},
    {"name": "Radish", "icon": "https://example.com/icons/radish.png"},
    {"name": "Tomato", "icon": "https://example.com/icons/tomato.png"}
]

# Custom styles
st.set_page_config(
    page_title="Vegetable Classifier",
    layout="centered",  # Set layout to centered
    page_icon="🍅",
)

def custom_css():
    st.markdown(
        """
        <style>
            .main {
                background-color: #f7f7f7;
                font-family: 'Arial', sans-serif;
                text-align: center;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stFileUploader {
                text-align: center;
            }
            /* Vegetable List Styling */
            .veg-list {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 20px;
                margin-top: 30px;
                padding: 0 20px;
            }
            .veg-item {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            .veg-item:hover {
                transform: translateY(-5px);
            }
            .veg-title {
                font-weight: bold;
                font-size: 16px;
                color: #4CAF50;
                text-align: center;
                margin-bottom: 10px;
            }
            .veg-icon {
                display: block;
                margin: 0 auto 10px;
                max-width: 50px;
                max-height: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

custom_css()

# Main title
st.markdown("<h1 style='color: #4CAF50;'>🍅 Vegetable Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload an image and let the AI predict the vegetable type!</p>", unsafe_allow_html=True)

# Display the vegetable list with icons in a grid layout
st.markdown("### Available Vegetables")
st.markdown(
    "<div class='veg-list'>"
    + "".join([
        f"<div class='veg-item'>"
        f"<img class='veg-icon' src='{veg['icon']}' alt='{veg['name']} icon'>"
        f"<p class='veg-title'>{veg['name']}</p>"
        f"</div>"
        for veg in class_labels
    ])
    + "</div>",
    unsafe_allow_html=True,
)

# Upload file
uploaded_file = st.file_uploader("Upload a Vegetable Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = class_labels[np.argmax(predictions)]['name']
    confidence = np.max(predictions)

    # Display results
    st.markdown(
        f"<h3 style='color: #4CAF50;'>Prediction: {predicted_class}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color: gray;'>Confidence: {confidence:.2f}</p>",
        unsafe_allow_html=True,
    )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Made by M. Bagus Prayogi</p>",
    unsafe_allow_html=True,
)
