import os
import tempfile
import urllib
from cvtk.ml.data import DataLabel
from cvtk.ml.mmdetutils import MMDETCORE
import label_studio_ml
import label_studio_ml.model
import label_studio_ml.api


class MLBASE(label_studio_ml.model.LabelStudioMLBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # laebl config
        self.LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = os.getenv('LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT')
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

        # model settings
        self.temp_dpath = tempfile.mkdtemp()
        self.datalabel = DataLabel("__DATALABEL__")
        self.model = MMDETCORE(self.datalabel, "__MODELCFG__", "__MODELWEIGHT__", workspace=self.temp_dpath)
        self.version = '0.0.0'


    def __del__(self):
        import shutil
        shutil.rmtree(self.temp_dpath)
        


    def fit(self, *args, **kwargs):
        return {'labels': '', 'model_file': ''}
    

    def predict(self, tasks, **kwargs):
        target_images = []
        for task in tasks:
            target_images.append(self.__abspath(task['data']['image']))

        pred_outputs = self.model.inference(target_images)
        for i, pred_output in enumerate(pred_outputs):
            pred_outputs[i] = self.__convert(pred_output)
        
        return pred_outputs

        
    def __abspath(self, filename):
        filename = filename.replace('/data/local-files/?d=', '')
        if self.LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is not None:
            filename = os.path.join(self.LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT, filename)
        return urllib.parse.unquote(filename)
    

    def __convert(self, im):
        obj_instances = []
        for ann in im.annotations:
            # skip low score annotations
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



app = label_studio_ml.api.init_app(
    model_class=MLBASE,
    model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    redis_port=os.environ.get('REDIS_PORT', 6379)
)


"""
Example:

gunicorn --bind 0.0.0.0:8600  main:app --reload
"""

