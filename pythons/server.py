from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, request, json, jsonify
from flask_cors import CORS
import sys
import utils
import numpy as np
from PIL import Image
import base64
from NeuralNetModel import NeuralNetModel
from ConvNetModel import ConvNetModel
from add_to_db import add_to_db


app = Flask(__name__)
CORS(app)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@app.route("/")
def index():
    return "Use /action"

# http://localhost:4002/api/process_image?dataUrl=DATA_URL
@app.route("/process_image")
def process_image():
    dataUrl = request.args.get('dataUrl', '')
    size = (28, 28)
    np_image = utils.data_url_to_arr(dataUrl, size)
    pixel_arr = utils.np_image_to_array(np_image).tolist()

    TRAIN_LIMIT = 60000
    TEST_LIMIT = 5000

    # Regular Neural Net:

    mnist_normal_model = NeuralNetModel(nn_type="normal", source="mnist", train_limit=TRAIN_LIMIT, test_limit=TEST_LIMIT)
    mnist_normal_model.train()
    mnist_normal_guess, mnist_normal_one_hot_result = mnist_normal_model.run_model(pixel_arr)

    web_canvas_normal_model = NeuralNetModel(nn_type="normal", source="web_canvas", train_limit=TRAIN_LIMIT, test_limit=TEST_LIMIT)
    web_canvas_normal_model.train()
    web_canvas_normal_guess, web_canvas_normal_one_hot_result = web_canvas_normal_model.run_model(pixel_arr)

    # utils.save_image(dataUrl)

    # Convolutional Neural Net:
    mnist_conv_model = ConvNetModel(nn_type="conv", source="mnist", train_limit=TRAIN_LIMIT, test_limit=TEST_LIMIT)
    mnist_conv_model.train()
    mnist_conv_guess, mnist_conv_one_hot_result = mnist_conv_model.run_model(pixel_arr)

    # Convolutional Neural Net:
    web_canvas_conv_model = ConvNetModel(nn_type="conv", source="web_canvas", train_limit=TRAIN_LIMIT, test_limit=TEST_LIMIT)
    web_canvas_conv_model.train()
    web_canvas_conv_guess, web_canvas_conv_one_hot_result = web_canvas_conv_model.run_model(pixel_arr)

    # mnist_guess = 1
    # mnist_one_hot_result = [1, 2, 3, 4, 5]
    # web_canvas_guess = 1
    # web_canvas_one_hot_result = [1, 2, 3, 4, 5]

    mnist_guess = mnist_conv_guess
    mnist_one_hot_result = mnist_conv_one_hot_result
    web_canvas_guess = web_canvas_conv_guess
    web_canvas_one_hot_result = web_canvas_conv_one_hot_result

    data = [
        {
            'source': 'mnist',
            'nn_type': 'normal',
            'prediction': mnist_normal_guess,
            'oneHotResult': mnist_normal_one_hot_result
        },
        {
            'source': 'web_canvas',
            'nn_type': 'normal',
            'prediction': web_canvas_normal_guess,
            'oneHotResult': web_canvas_normal_one_hot_result
        },
        {
            'source': 'mnist',
            'nn_type': 'conv',
            'prediction': mnist_conv_guess,
            'oneHotResult': mnist_conv_one_hot_result
        },
        {
            'source': 'web_canvas',
            'nn_type': 'conv',
            'prediction': web_canvas_conv_guess,
            'oneHotResult': web_canvas_conv_one_hot_result
        },

    ]
    return jsonify(netStatistics=data)

@app.route("/add_training_image")
def add_training_image():
    dataUrl = request.args.get('dataUrl', '')
    label = request.args.get('label', '')
    size = (28, 28)
    np_image = utils.data_url_to_arr(dataUrl, size)
    features = utils.np_image_to_array(np_image).tolist()
    one_hot_label = utils.int_to_one_hot(label).tolist()

    source = "web_canvas"
    add_to_db(str(features), str(one_hot_label), source)
    eprint("yoyoyoyoyoyoyoyoyoy", type(features), label, source)
    return "200 OK"


@app.route("/train_models")
def train_model():
    mnist_model = NeuralNetModel(checkpoint_folder="/mnist", train_limit=5000, test_limit=1000)
    mnist_model.train()

    web_canvas_model = NeuralNetModel(checkpoint_folder="/web_canvas", train_limit=5000, test_limit=1000)
    web_canvas_model.train()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4002)
