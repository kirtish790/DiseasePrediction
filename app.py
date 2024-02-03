from flask import Flask, request, render_template, url_for
from flask import Flask, request, render_template, redirect
import io
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import os
from utils.disease import disease_dic
import requests
from PIL import Image
from utils.model import ResNet9
from bs4 import BeautifulSoup

app = Flask(__name__)

disease_classes = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Potato___Late_blight",
    "Potato___healthy",
]

disease_model_path = r'C:\Users\Admin\Downloads\crop disease prediction\models\plant-disease-model (1).pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    
    return prediction
@app.route('/', methods=['GET'])
def main_page():
    title = 'Harvestify - Disease Detection'
    return render_template('disease.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            prediction = BeautifulSoup(prediction, "html.parser").get_text()

            prediction = str(disease_dic[prediction])
            return render_template('disease_results.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

if __name__ == "__main__":
    app.run(debug=True)