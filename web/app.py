import os
import base64
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("model/plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        data=request.json
        if 'image' not in data:
            return jsonify({'error':'No image data found'}), 400

        image_data=data['image']
        # Decode the base64 image
        try:
            image_bytes=base64.b64decode(image_data)
        expect (ValueError,IOError) as e:
            return jsonify({'error':str(e)}), 400

        #Save the image to file
        filename='upload_image.png'
        image_path=os.path.join('static/uploads'.filename)
        with open(image_path, 'wb') as image_file:
            image_file.write(image_bytes)

        pred = prediction(file_path)
        print(pred,image_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return jsonify({'message':'Image uploaded successfully!','data':{'title':title , 'description':description , 'prevent':prevent, 'image_url':image_url , 'pred':pred ,'s_name':supplement_name , 's_image':supplement_image_url , 'buy_link':supplement_buy_link}}), 200

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
