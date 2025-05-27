from flask import Flask, request, render_template, jsonify
from script import predict_image
import os
from script import predict_image, encoder, support_set_path, device


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    file = request.files['file']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        label = predict_image(filepath, encoder=encoder, support_path=support_set_path, device=device)
        return jsonify({'prediction': label})
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)
