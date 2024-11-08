from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/malaria_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalization
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file temporarily
        file_path = 'static/uploads/uploaded_image.png'  # Adjust path as needed
        file.save(file_path)

        # Preprocess the uploaded image
        img = preprocess_image(file_path)

        # Use the loaded model to make a prediction
        prediction = model.predict(img)

        # Check the predicted class and interpret it correctly
        # Assuming model uses 0 for Infected (Parasitized) and 1 for Uninfected
        if prediction[0][0] > 0.5:
            result = "Uninfected"  # Prediction closer to 1 means Uninfected
        else:
            result = "Infected"  # Prediction closer to 0 means Infected

        # Delete the temporary file after prediction
        os.remove(file_path)

        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
