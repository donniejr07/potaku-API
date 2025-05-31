from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

app = Flask(__name__)
model = load_model('model/mobilenetv2_klasifikasi.h5')

LABELS = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']

def prepare_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)
        preds = model.predict(processed_image)
        class_idx = np.argmax(preds[0])
        result = LABELS[class_idx] if class_idx < len(LABELS) else str(class_idx)
        return jsonify({'class': result, 'confidence': float(np.max(preds[0]))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)