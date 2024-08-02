import os
import json
import pathlib
import hashlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
APP_STORAGE = os.path.join(APP_ROOT, 'storage')
APP_TEMP = os.path.join(APP_ROOT, 'tmp')
MODEL = MODULECORE('__DATALABEL__', '__MODELCFG__','__MODELWEIGHT__', workspace=APP_TEMP)
if not os.path.exists(APP_STORAGE):
    os.makedirs(APP_STORAGE)
if not os.path.exists(APP_TEMP):
    os.makedirs(APP_TEMP)

# application setup
app = FastAPI()
app.mount('/storage', StaticFiles(directory='storage'), name='storage')
templates = Jinja2Templates(directory=os.path.join(APP_ROOT, 'templates'))



@app.get('/')
def startup_page(request: Request):
    return templates.TemplateResponse('index.html',
                                      {'request': request, 'id': id})


@app.post('/api/inference')
async def inference(request: Request):
    form = await request.form()
    image_data = form['file']

    # save image to storage
    image = await image_data.read()
    image_fpath = save_image(image, image_data.filename)
    
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
        'image': os.path.join('storage', os.path.basename(image_fpath)),
        'annotations': [_ for _ in output[0].annotations]
    }
    for i in range(len(output['annotations'])):
        if 'mask' in output['annotations'][i] and output['annotations'][i]['mask'] is not None:
            output['annotations'][i]['polygons'] = []
            for contour in skimage.measure.find_contours(np.array(output['annotations'][i]['mask']), 0.5):
                output['annotations'][i]['polygons'].append([[c[1], c[0]] for c in contour.tolist()])
            output['annotations'][i]['mask'] = None # data is too large
#%CVTK%# ENDIF


    return JSONResponse(content = {'data': output})


def save_image(im, image_fname):
    token = hashlib.md5(im).hexdigest()
    im_fpath = os.path.join(APP_STORAGE, token + os.path.splitext(image_fname)[1])
    if not os.path.exists(im_fpath):
        with open(im_fpath, 'wb') as fh:
            fh.write(im)
    return im_fpath




"""
Example:

uvicorn main:app --host 0.0.0.0 --port 8080 --reload
"""

