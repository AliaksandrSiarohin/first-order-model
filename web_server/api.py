# Based in https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/

import os
from flask import request
from werkzeug.utils import secure_filename
from flask_api import FlaskAPI

app = FlaskAPI(__name__)

UPLOAD_FOLDER = './uploads'

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
        status = request.files['source']

        error = verifyFile(request, 'driving', {'mp4'})
        if error: return error
        driving = request.files['driving']

        filename = secure_filename(status.filename)
        status.save(os.path.join(UPLOAD_FOLDER, filename))

        filename = secure_filename(driving.filename)
        driving.save(os.path.join(UPLOAD_FOLDER, filename))

        return {'status': "Sucess"}, 200

