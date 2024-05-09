from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
from ocr import Enhanced_OCR_CNN  # Make sure this import points to your model's definition
import string
import os
import logging
from google.cloud import logging as google_logging
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/upload": {"origins": "https://test-wpek6upsvq-nw.a.run.app/"}})

@app.route('/')
def home():
    # Make sure the path is correct
    return redirect(url_for('static', filename='main.html'))

# Load your trained model
model = Enhanced_OCR_CNN()
model.load_state_dict(torch.load('model_ocr.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transformations = transforms.Compose([
    transforms.Resize((28, 28)),          # Resize the image to 28x28 pixels
    transforms.ToTensor(),                # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

# Define label map as in test.py
label_map = list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)

def decode_label(label_index):
    """
    Decode the predicted class index into a character.
    """
    if label_index < len(label_map):
        return label_map[label_index]
    else:
        return "Character not recognized"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream).convert('L')  # Convert to grayscale
        image_tensor = transformations(image).unsqueeze(0)  # Apply transformations and add batch dimension

        with torch.no_grad():
            prediction = model(image_tensor)
            _, predicted_idx = torch.max(prediction, 1)
            predicted_char = decode_label(predicted_idx.item())

        return jsonify({'text': predicted_char})
    except Exception as e:
        app.logger.error(f"Error processing the image: {str(e)}")
        return jsonify({'error': str(e)}), 500

#client = google_logging.Client()
#client.setup_logging()

#logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)