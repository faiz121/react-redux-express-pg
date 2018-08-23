from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, request, json, jsonify
from flask_cors import CORS
import sys
import utils
import numpy as np
from PIL import Image
import base64
from NeuralNetModel import NeuralNetModel
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

    model = NeuralNetModel()
    # model.train()
    guess, one_hot_result = model.run_model(pixel_arr)
    utils.save_image(dataUrl)
    return jsonify(guess=json.dumps(guess.tolist()), dataUrl=dataUrl, one_hot_result=json.dumps(one_hot_result))

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4002)
