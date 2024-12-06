from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the Keras model and convert to TFLite
model = keras.models.load_model('cnn_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('cnn_modell.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='cnn_modell.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print("Input details:", input_details)
# print("Output details:", output_details)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read the image file and preprocess it
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # Resize the image to 200x200 (model's expected input size)
    img = img.resize((200, 200))
    
    # Convert to numpy array
    img_array = img_to_array(img)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('OUTPUT DATA')
    print(output_data[0][0])
    # Interpret the result
    if output_data[0][0] > 0.5:
        prediction = "Class 1 (Curly)"
    else:
        prediction = "Class 0 (Straight)"
    
    return jsonify({'prediction': prediction, 'probability': str(output_data[0][0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)



# img_path = "/workspaces/llm-zoomcamp/serverless/data/test/curly/1a9dbe23a0d95f1c292625960e4509184.jpg"  # Replace with the actual image path
# img = load_img(img_path, target_size=(200, 200))  # Resize the image
# img_array = img_to_array(img)  # Convert to numpy array
# img_array = img_array / 255.0  # Normalize pixel values
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# # Set the input tensor
# interpreter.set_tensor(input_details[0]['index'], img_array)

# # Run inference
# interpreter.invoke()

# # Get the output tensor
# output_data = interpreter.get_tensor(output_details[0]['index'])

# # Interpret the result
# if output_data[0][0] > 0.5:
#     print("Prediction: Class 1 (Curly)")
# else:
#     print("Prediction: Class 0 (Straight)")





