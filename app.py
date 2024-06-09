import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('model.pkl')  # Update with your actual model file path

# Streamlit app title
st.title("Handwritten Letter Recognition")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Processing...")

    # Convert to grayscale
    image = image.convert('L')

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize the array (Optional: Depending on your model requirements, e.g., scaling to [0, 1])
    image_array = image_array / 255.0  # If normalization was part of training

    # Reshape the array to match the model's expected input shape
    image_array = image_array.reshape(1, 28, 28)  # Example for CNN, adjust if needed

    # Predict the letter using the model
    prediction = model.predict(image_array)

    # Get the predicted letter (Assuming the model outputs the class index)
    predicted_letter = chr(prediction.argmax())[0]

    # Display the prediction
    st.write(f"Predicted Letter: {predicted_letter}")

