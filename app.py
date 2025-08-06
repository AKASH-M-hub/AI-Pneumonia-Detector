import os
import numpy as np
import cv2  # Using OpenCV for image reading
from flask import Flask, request, render_template, send_from_directory
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FILE = 'pneumonia_model_quant.tflite'

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image with OpenCV
    img = cv2.imread(image_path)
    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image
    img = cv2.resize(img, target_size)
    # Add a batch dimension and convert to float32
    img_array = np.expand_dims(img, axis=0).astype(np.float32)
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

            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)

            # Set the value of the input tensor
            interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

            # Run the inference
            interpreter.invoke()

            # Get the prediction
            prediction = interpreter.get_tensor(output_details[0]['index'])

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