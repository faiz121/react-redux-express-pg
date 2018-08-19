from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, request, json, jsonify
from flask_cors import CORS
import sys
import utils
import numpy as np
from PIL import Image
import base64
from NeuralNetModel import NeuralNetModel


app = Flask(__name__)
CORS(app)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@app.route("/")
def index():
    return "Use /api/action"

# http://localhost:4002/api/process_image?dataUrl=DATA_URL
@app.route("/process_image")
def process_image():
    dataUrl = request.args.get('dataUrl', '')
    size = (28, 28)
    np_image = utils.data_url_to_arr(dataUrl, size)
    pixel_arr = utils.np_image_to_array(np_image).tolist()

    model = NeuralNetModel()
    # model.train()
    model.run_model(pixel_arr)
    utils.save_image(dataUrl)
    return jsonify({ 'dataUrl': dataUrl })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4002)
