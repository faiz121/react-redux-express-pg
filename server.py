from __future__ import division, print_function, absolute_import
from flask import Flask, render_template, request, json, jsonify
import sys

app = Flask(__name__)
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@app.route("/")
def index():
    return "Use /api/action"

# http://localhost:4002/api/process_image?dataUrl=DATA_URL
@app.route("/api/process_image")
def process_image():
    dataUrl = request.args.get('dataUrl', '')
    return jsonify({ 'dataUrl': dataUrl })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4002)
