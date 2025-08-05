import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from serverless_wsgi import handle  # Import the handler

app = Flask(__name__, static_folder='../../static', template_folder='../../static/templates')

# Define paths relative to the project root
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'pneumonia_detector_model.h5')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'uploads')

# Load the model
model = load_model(MODEL_PATH)

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# (The preprocess_image function remains the same)
def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            preprocessed_image = preprocess_image(filepath)
            prediction = model.predict(preprocessed_image)

            result_data = {}
            if prediction[0][0] < 0.5:
                result_data['text'] = "Prediction: NORMAL"
                result_data['confidence'] = f"Confidence: {100*(1-prediction[0][0]):.2f}%"
                result_data['class'] = 'success'
            else:
                result_data['text'] = "Prediction: PNEUMONIA"
                result_data['confidence'] = f"Confidence: {100*prediction[0][0]:.2f}%"
                result_data['class'] = 'danger'

            return render_template('result.html', result=result_data, image_name=file.filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# This is the handler that Netlify will use
def handler(event, context):
    return handle(app, event, context)