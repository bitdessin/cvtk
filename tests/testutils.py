import json
import os
import subprocess


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(TESTS_DIR, 'outputs')


def _test_path(*parts):
    return os.path.join(TESTS_DIR, *parts)

__cls_data = {
    'label': _test_path('data', 'fruits', 'label.txt'),
    'all': _test_path('data', 'fruits', 'all.txt'),
    'train': _test_path('data', 'fruits', 'train.txt'),
    'valid': _test_path('data', 'fruits', 'valid.txt'),
    'test': _test_path('data', 'fruits', 'test.txt'),
    'samples': _test_path('data', 'fruits', 'images'),
    'sample': _test_path('data', 'fruits', 'images', '14c6e557.jpg'),
}
__det_data = {
    'label': _test_path('data', 'strawberry', 'label.txt'),
    'train': _test_path('data', 'strawberry', 'train', 'bbox.json'),
    'valid': _test_path('data', 'strawberry', 'valid', 'bbox.json'),
    'test': _test_path('data', 'strawberry', 'test', 'bbox.json'),
    'samples': _test_path('data', 'strawberry', 'test', 'images'),
    'test_result': _test_path('data', 'strawberry', 'test', 'test_outputs.bbox.json'),
}
__segm_data = {
    'label': _test_path('data', 'strawberry', 'label.txt'),
    'train': _test_path('data', 'strawberry', 'train', 'segm.json'),
    'valid': _test_path('data', 'strawberry', 'valid', 'segm.json'),
    'test': _test_path('data', 'strawberry', 'test', 'segm.json'),
    'samples': _test_path('data', 'strawberry', 'test', 'images'),
    'test_result': _test_path('data', 'strawberry', 'test', 'test_outputs.segm.json'),
}
data = {
    'cls': __cls_data,
    'det': __det_data,
    'segm': __segm_data
}


def set_ws(dpath):
    dpath = os.path.join(OUTPUTS_DIR, dpath)
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    return dpath


def run_cmd(cmd):
    cmd = [str(_) for _ in cmd]
    print('\nCOMMAND -----------------------------------------')
    print(' '.join(cmd))
    print('-------------------------------------------------\n')
    output = subprocess.run(cmd, cwd=TESTS_DIR)
    if output.returncode != 0:
        raise Exception('Error: {}'.format(output.returncode))


class COCO():
    def __init__(self, file_path):
        with open(file_path) as infh:
            self.data = json.load(infh)

        self.images = set([_['file_name'] for _ in self.data['images']])
        self.annotations = set([_['id'] for _ in self.data['annotations']])
        self.categories = [_['name'] for _ in sorted(self.data['categories'], key=lambda x: x['id'])]

        
