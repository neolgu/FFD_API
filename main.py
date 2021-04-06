import io
import json
import os
import base64
import re

from PIL import Image
import flask
from flask import Flask, jsonify, request
import torch

from network.conf import ConfModel
from data.transform import xception_data_transforms

app = Flask(__name__)

# Choose CPU, GPU
use_gpu = False  # torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# Load model

net = ConfModel(num_classes=2)
net.to(device)
net.load_state_dict(torch.load("checkpoint\\conf\\17.tar"))
net.eval()

# print(net)


# Transform input into the form our model expects
def transform_image(image_bytes):
    transform = xception_data_transforms['test']
    image = Image.open(image_bytes).convert('RGB')
    tensor_img = transform(image)
    tensor_img.unsqueeze_(0)  # batch 1
    return tensor_img


# Get a prediction
def get_prediction(input_tensor):
    outputs = net.forward(input_tensor)
    _, preds = outputs.max(1)
    print(outputs)
    return preds


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.form['img']
        if file is not None:
            temp = re.sub('^data:image/.+;base64,', '', file)
            file = io.BytesIO(base64.b64decode(temp))
            tensor_img = transform_image(file)
            prediction = get_prediction(tensor_img)
            response = jsonify({'pred_result': int(prediction)})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost/')
            return response
        else:
            response = jsonify({'msg': 'File Error. Please Check file is image.'})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost/')
            return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
