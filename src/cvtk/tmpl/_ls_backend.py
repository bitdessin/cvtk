import os
import argparse
import urllib
import shutil
import datetime
import tempfile
import PIL.Image
import torch
from cvtk.ml.data import DataLabel
from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, ModuleCore
import label_studio_ml
import label_studio_ml.model
import label_studio_ml.api
import label_studio_tools


class MLBASE(label_studio_ml.model.LabelStudioMLBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # basic params
        self.LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL', None)
        self.LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = os.getenv('LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT', None)
        self.LABEL_STUDIO_BASE_DATA_DIR = os.getenv('LABEL_STUDIO_BASE_DATA_DIR', None)

        # label config
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

        # model settings
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.temp_dpath = tempfile.mkdtemp()
        self.datalabel = DataLabel("__DATALABEL__")
        self.model = ModuleCore(self.datalabel, "__MODELCFG__", "__MODELWEIGHT__", workspace=self.temp_dpath)
        self.model.to(self.device)
        self.version = '0.0.0'


    def __del__(self):
        shutil.rmtree(self.temp_dpath)
        

    def fit(self, tasks, workdir=None, **kwargs):
        event = kwargs.get('event', None)
        if event == 'START_TRAINING':
            raise NotImplementedError('Training is not implemented yet.')
        return {'labels': '', 'model_file': '', 'version': ''}
    

    def predict(self, tasks, context, **kwargs):
        self.mdoel.eval()

        target_images = []
        for task in tasks:
            target_images.append(self.__get_image(task))

        with torch.no_grad():
            outputs = self.model.inference(target_images)
                
        return [self.__detoutput2lsjson(o) for o in outputs]


    def __get_image(self, task):
        im_fpath = task['data']['image']
        if '/data/local-files/?d=' in im_fpath:
            im_fpath = label_studio_tools.core.utils.io.get_local_path(task['data']['image'], task_id=task['id'])
        elif '/data/upload/' in im_fpath:
            im_fpath = im_fpath.replace('/data/', '')
            im_fpath = os.path.join(self.LABEL_STUDIO_BASE_DATA_DIR, 'media', im_fpath)
        else:
            print('Warning: cannot recognize the file path format.')
            print(im_fpath)
            print('-----------')
        return urllib.parse.unquote(im_fpath)


    def __detoutput2lsjson(self, im):
        obj_instances = []
        for ann in im.annotations:
            if ann['score'] < 0.5:
                continue

            x1, y1, x2, y2 = ann['bbox']
            w = float((x2 - x1) / im.width * 100)
            h = float((y2 - y1) / im.height * 100)
            x = float(x1 / im.width * 100)
            y = float(y1 / im.height * 100)

            obj_instances.append({
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'rectanglelabels',
                'original_width': im.width,
                'original_height': im.height,
                'value': {
                    'rectanglelabels': [ann['label']],
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'score': ann['score'],
                }
            })

        return {
            'result': obj_instances,
            'score': 1.0,
            'model_version': self.version,
        }




if __name__ == '__main__':
    if not os.getenv('LABEL_STUDIO_BASE_DATA_DIR'):
        Warning('Environment variable "LABEL_STUDIO_BASE_DATA_DIR" is not defined. It is required to treat images uploaded to Label Studio via API or web browser.')
    if not os.getenv('LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'):
        Warning('Environment variable "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" is not defined. It is required to treat images synced from local storage.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    app = label_studio_ml.api.init_app(
        model_class=MLBASE,
        model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir'),
    )
    app.run(host=args.host, port=args.port, debug=True)


"""
Example:

    python mlbackend.py
"""