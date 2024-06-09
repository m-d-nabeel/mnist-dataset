from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')  # Update with your actual model file path

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image):
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize the array (Optional: Depending on your model requirements)
    image_array = image_array / 255.0  # Normalize to [0, 1]

    # Reshape the array to match the model's expected input shape
    image_array = image_array.reshape(1, 28, 28)  # Example for CNN, adjust if needed
    
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser may submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Open the image file
            image = Image.open(file)
            image_array = process_image(image)
            predicted_letter = model.predict(image_array.reshape(1, 28, 28)).argmax(axis=1)[0]
            
            return render_template('index.html', prediction=predicted_letter)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
