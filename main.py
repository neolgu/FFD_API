import io
import json
import os

from PIL import Image
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
    image = Image.open(image_bytes)
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
        file = request.files['file']
        if file is not None:
            tensor_img = transform_image(file)
            prediction = get_prediction(tensor_img)
            return jsonify({'pred_result': int(prediction)})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
