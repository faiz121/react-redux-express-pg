from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, request, json, jsonify
from flask_cors import CORS
import sys
from pythons import utils
import numpy as np
from PIL import Image
import base64


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
    pixel_arr = utils.data_url_to_arr(dataUrl, size)
    utils.save_image(dataUrl)
    return jsonify({ 'dataUrl': dataUrl })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4002)
