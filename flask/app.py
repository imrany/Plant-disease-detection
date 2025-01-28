import os
import base64
from flask import Flask, redirect, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import CNN
import numpy as np
import torch
import pandas as pd
import io

print(torch.__version__)
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load("model/plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    
    # If the image has an alpha channel (4 channels), convert it to RGB (3 channels)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Define the transformation: Resize to 224x224 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),          # Convert image to PyTorch tensor
    ])

    # Apply the transformation to the image
    input_data = transform(image)

    # Add a batch dimension (1, 3, 224, 224)
    input_data = input_data.unsqueeze(0)
    
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit_json', methods=['POST'])
def submitJson():
    global image_path
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        try:
            image = Image.open(io.BytesIO(file.read()))
            image_path = os.path.join('static/uploads', file.filename)
            image.save(image_path)
        except (ValueError, IOError) as e:
            return jsonify({'error': str(e)}), 400
    elif request.json and 'image' in request.json:
        image_data = request.json['image']
        try:
            image_bytes = base64.b64decode(image_data)
            image_path = os.path.join('static/uploads', 'upload_image.png')
            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)
        except (ValueError, IOError) as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No image data found'}), 400

    pred = prediction(image_path)
    print(pred, image_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    return jsonify({
        'message': 'Image uploaded successfully!',
        'data': {
            'title': title,
            'description': description,
            'prevent': prevent,
            'image_url': image_url,
            'pred': int(pred),
            's_name': supplement_name,
            's_image': supplement_image_url,
            'buy_link': supplement_buy_link
        }
    }), 200

@app.route('/submit', methods=['POST','GET'])
def submit():
    global image_path
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template(
                'result.html',
                title='Error',
                error='No selected file'
            ), 400
        try:
            image = Image.open(io.BytesIO(file.read()))
            image_path = os.path.join('static/uploads', file.filename)
            image.save(image_path)
        except (ValueError, IOError) as e:
            return jsonify({'error': str(e)}), 400
    elif request.json and 'image' in request.json:
        image_data = request.json['image']
        try:
            image_bytes = base64.b64decode(image_data)
            image_path = os.path.join('static/uploads', 'upload_image.png')
            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)
        except (ValueError, IOError) as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No image data found'}), 400

    pred = prediction(image_path)
    print(pred, image_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    return render_template('result.html', **{
        'title': title,
        'description': description,
        'prevent': prevent,
        'image_url': image_url,
        'pred': int(pred),
        's_name': supplement_name,
        's_image': supplement_image_url,
        'buy_link': supplement_buy_link
    }), 200

@app.route('/supplements', methods=['GET', 'POST'])
def supplements():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
