# Based in https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/

import os
from flask import request
from werkzeug.utils import secure_filename
from flask_api import FlaskAPI
import processing as p
import numpy as np

app = FlaskAPI(__name__)

UPstream_FOLDER = './uploads'

def allowed_file(filename, extension = {'png', 'jpg', 'jpeg'}):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extension

def verifyFile(request, parmname, allowed_extensions):
    if parmname not in request.files:
        return {'error': f'{parmname} is required'}, 400

    file = request.files[parmname]

    if file.filename == '':
        return {'error': f'{parmname} is empty'}, 400

    if not allowed_file(file.filename, allowed_extensions):
        return {'error': f'Invalid {parmname} format'}, 400


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        error = verifyFile(request, 'source', {'jpg', 'jpeg', 'png'})
        if error: return error
        status_req = request.files['source']

        error = verifyFile(request, 'driving', {'mp4'})
        if error: return error
        driving_req = request.files['driving']

        status = p.img2opencv(status_req)
        status_req.close()
        status = p.crop_resize(status)
        # cv2.imwrite('uploads/image.png',status)

        driving = p.vid2opencv(driving_req)
        driving_req.close()
        driving = p.v_crop_resize(driving)

        # https://medium.com/analytics-vidhya/receive-or-return-files-flask-api-8389d42b0684
        return {'status': "Sucess", 'shape': np.array(list(driving)).shape}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, threaded = True, debug = True)
