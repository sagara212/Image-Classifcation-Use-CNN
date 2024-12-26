import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load model
model = load_model('model_CNN.keras')

# Class labels and additional information
class_labels = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

vegetable_info = {
    "Broccoli": "Rich in vitamins K and C, broccoli supports bone health and immune function.",
    "Capsicum": "High in vitamin C, capsicum helps improve skin health and boost immunity.",
    "Bottle_Gourd": "Low in calories and high in fiber, great for digestion and hydration.",
    "Radish": "Contains antioxidants and supports detoxification.",
    "Tomato": "A great source of lycopene, an antioxidant linked to heart health.",
    "Brinjal": "Rich in fiber and low in calories, helps control blood sugar levels.",
    "Pumpkin": "Packed with vitamin A, supports vision and skin health.",
    "Carrot": "Excellent source of beta-carotene, essential for eye health.",
    "Papaya": "Rich in enzymes that aid digestion and high in vitamin C.",
    "Cabbage": "Contains antioxidants and supports gut health.",
    "Bitter_Gourd": "Helps regulate blood sugar levels and is high in vitamins.",
    "Cauliflower": "Rich in fiber and B-vitamins, promotes brain health.",
    "Bean": "Good source of plant-based protein and supports heart health.",
    "Cucumber": "Hydrating and cooling, contains antioxidants and vitamins.",
    "Potato": "Rich in potassium, supports energy and muscle function."
}

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
            .veg-description {
                font-size: 14px;
                color: gray;
                text-align: center;
                margin-bottom: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

custom_css()

# Main title
st.markdown("<h1 style='color: #4CAF50;'>🍅 Vegetable Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload an image and let the AI predict the vegetable type!</p>", unsafe_allow_html=True)

# Display the vegetable list in a grid layout
st.markdown("### Available Vegetables")
st.markdown(
    "<div class='veg-list'>"
    + "".join([f"<div class='veg-item'><p class='veg-title'>{veg}</p></div>" for veg in class_labels])
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
    predicted_class = class_labels[np.argmax(predictions)]
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

    # Display additional information
    if predicted_class in vegetable_info:
        st.markdown(f"### Fun Fact about {predicted_class}:")
        st.write(vegetable_info[predicted_class])

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Made by M. Bagus Prayogi</p>",
    unsafe_allow_html=True,
)
