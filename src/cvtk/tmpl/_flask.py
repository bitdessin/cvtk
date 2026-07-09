import os
import sys
import json
import pathlib
import hashlib
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import skimage.measure

APP_ROOT = pathlib.Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from __MODULE_IMPORT__ import __MODULE_CLASS__ as MODULECORE


# application variables
APP_STORAGE = os.path.join(APP_ROOT, 'static', 'storage')
APP_TEMP = os.path.join(APP_ROOT, 'tmp')

#%CVTK%# IF TASK=cls,det,segm
#%CVTK%# IF BACKEND=torch
MODEL = MODULECORE(os.path.join(APP_ROOT, '__DATALABEL__'),
                   '__MODELCFG__',
                   os.path.join(APP_ROOT, '__MODELWEIGHT__'), workspace=APP_TEMP)
#%CVTK%# ENDIF

#%CVTK%# IF BACKEND=mmdet
MODEL = MODULECORE(os.path.join(APP_ROOT, '__DATALABEL__'),
                   os.path.join(APP_ROOT, '__MODELCFG__'),
                   os.path.join(APP_ROOT, '__MODELWEIGHT__'), workspace=APP_TEMP)
#%CVTK%# ENDIF
#%CVTK%# ENDIF
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
                output = MODEL.inference(image_fpath, format='pandas')
                # Convert DataFrame row to list of dicts with label and prob, sorted by prob
                if len(output) > 0:
                    row = output.iloc[0]
                    output_table = [{'label': label, 'prob': float(prob)} for label, prob in row.items()]
                    output = sorted(output_table, key=lambda x: x['prob'], reverse=True)
                else:
                    output = []
#%CVTK%# ENDIF

#%CVTK%# IF TASK=det,segm
                output = MODEL.inference(image_fpath, cutoff=0.5)
                # Get first result (handle both list and ImageDataset by iterating)
                first_result = next(iter(output))
                output = {
                    'image': os.path.join('static', 'storage', os.path.basename(image_fpath)),
                    'annotations': []
                }
                # Convert InstanceAnnotation objects to dictionaries and process masks
                for ann in first_result.annotations:
                    ann_dict = ann.to_dict(segm_format='mask')
                    # Extract polygons from mask if available
                    if ann_dict['segm'] is not None:
                        ann_dict['polygons'] = []
                        try:
                            for contour in skimage.measure.find_contours(np.array(ann_dict['segm']), 0.5):
                                ann_dict['polygons'].append([[c[1], c[0]] for c in contour.tolist()])
                        except Exception:
                            pass
                        ann_dict['segm'] = None  # data is too large
                    output['annotations'].append(ann_dict)
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

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 gunicorn --bind 0.0.0.0:8600  main:app --reload
"""

