#!/usr/bin/env python

import flask
from flask import request, jsonify
from os import path, environ
from raven.contrib.flask import Sentry
import cv2
import numpy as np
import requests

from auth import token_required
from lib.detection_model import load_net, detect

THRESH = 0.08  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2

app = flask.Flask(__name__)

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

# Sentry
sentry = None
if environ.get('SENTRY_DSN'):
    sentry = Sentry(app, dsn=environ.get('SENTRY_DSN'))

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main, meta_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.weights'), path.join(model_dir, 'model.meta'))

def get_image_array_from_args():
    resp = requests.get(request.args['img'], stream=True, timeout=(1, 5))
    resp.raise_for_status()
    return np.array(bytearray(resp.content), dtype=np.uint8)

def get_image_array_from_file():
    return np.array(bytearray(request.files['img'].read()), dtype=np.uint8)

def get_image():
    if request.method == 'GET':
        img_array = get_image_array_from_args()
    elif request.method == 'POST':
        img_array = get_image_array_from_file()
    return cv2.imdecode(img_array, -1)

@app.route('/p/', methods=['GET', 'POST'])
@token_required
def get_p():
    if request.method == 'GET' and 'img' not in request.args:
        app.logger.warn("Invalid request params: {}".format(request.args))
    try:
        img = get_image()
        detections = detect(net_main, meta_main, img, thresh=THRESH)
        return jsonify({'detections': detections})
    except:
        if sentry:
            sentry.captureException()

    return jsonify({'detections': []})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
