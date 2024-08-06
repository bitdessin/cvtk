import os
import json
import pathlib
import hashlib
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import skimage.measure
#%CVTK%# IF TASK=cls
from cvtk.ml.torchutils import CLSCORE as MODULECORE
#%CVTK%# ENDIF
#%CVTK%# IF TASK=det,segm
from cvtk.ml.mmdetutils import MMDETCORE as MODULECORE
#%CVTK%# ENDIF

# application variables
APP_ROOT = pathlib.Path(__file__).resolve().parent
APP_STORAGE = os.path.join(APP_ROOT, 'static', 'storage')
APP_TEMP = os.path.join(APP_ROOT, 'tmp')
MODEL = MODULECORE('__DATALABEL__', '__MODELCFG__','__MODELWEIGHT__', workspace=APP_TEMP)
if not os.path.exists(APP_STORAGE):
    os.makedirs(APP_STORAGE)
if not os.path.exists(APP_TEMP):
    os.makedirs(APP_TEMP)

# application setup
app = Flask(__name__,
            static_folder=os.path.join(APP_ROOT, 'static'),
            static_url_path='/static')
app.config['UPLOAD_FOLDER'] = APP_STORAGE
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


@app.route('/', methods=['GET'])
def startup_page():
    return render_template('index.html')


@app.route('/api/inference', methods=['POST'])
def inference():
    output = None
    if request.method == 'POST':
        if 'file' in request.files:
            image_fpath = save_image(request.files['file'])
            if image_fpath is not None:

#%CVTK%# IF TASK=cls
                output = MODEL.inference(image_fpath, value='prob+label', format='pandas')
                output = json.loads(output.to_json(orient='records'))
                output = output[0]
                del output['prediction']
                output_table = []
                for cl, prob in output.items():
                    output_table.append({'label': cl, 'prob': float(prob)})
                output = sorted(output_table, key=lambda x: x['prob'], reverse=True)
#%CVTK%# ENDIF

#%CVTK%# IF TASK=det,segm
                output = MODEL.inference(image_fpath, cutoff=0.5)
                output = {
                    'image': os.path.join('static', 'storage', os.path.basename(image_fpath)),
                    'annotations': [_ for _ in output[0].annotations]
                }
                for i in range(len(output['annotations'])):
                    if 'mask' in output['annotations'][i] and output['annotations'][i]['mask'] is not None:
                        output['annotations'][i]['polygons'] = []
                        for contour in skimage.measure.find_contours(np.array(output['annotations'][i]['mask']), 0.5):
                            output['annotations'][i]['polygons'].append([[c[1], c[0]] for c in contour.tolist()])
                        output['annotations'][i]['mask'] = None # data is too large
#%CVTK%# ENDIF

    return jsonify({'data': output})



def save_image(req_file):
    im_fpath = None

    im = req_file.read()
    im_hash = hashlib.md5(im).hexdigest()

    f_ext = os.path.splitext(secure_filename(req_file.filename))[1]
    if f_ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.heic']:
        im_fpath = os.path.join(app.config['UPLOAD_FOLDER'], im_hash + f_ext)
        with open(im_fpath, 'wb') as fh:
            fh.write(im)

    return im_fpath



"""
Example:

gunicorn --bind 0.0.0.0:8600  main:app --reload
"""

