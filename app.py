from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib
import io
import base64

app = Flask(__name__)

# Load the SVM model
model = joblib.load('model.pkl')

def preprocess_image(uploaded_file):
    # Open the image
    image = Image.open(uploaded_file)
    
    # Resize the image to the expected dimensions
    resized_image = image.resize((64, 64))

    # Convert the resized image to an array
    img_array = np.asarray(resized_image)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Flatten the image
    flattened_image = img_array.flatten()

    # Ensure the number of features matches the model's expectation (4096 in this case)
    flattened_image = flattened_image[:4096]

    return flattened_image, resized_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    uploaded_file = request.files['image']
    
    # Preprocess the image
    processed_image, resized_image = preprocess_image(uploaded_file)

    # Make predictions using the loaded model
    prediction = model.predict([processed_image])

    # Decode the prediction (0 for cat, 1 for dog)
    result = "Cat" if prediction == 0 else "Dog"

    # Convert the resized image to base64 for display
    img_data = io.BytesIO()
    resized_image.save(img_data, format='JPEG')
    img_data = base64.b64encode(img_data.getvalue()).decode('utf-8')

    # Return the prediction result and the base64 encoded image for display
    return render_template('index.html', result=result, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
