import json
import os
import subprocess

__cls_data = {
    'label': './data/fruits/label.txt',
    'all': './data/fruits/all.txt',
    'train': './data/fruits/train.txt',
    'valid': './data/fruits/valid.txt',
    'test': './data/fruits/test.txt',
    'samples': './data/fruits/images',
    'sample': 'data/fruits/images/14c6e557.jpg',
}
__det_data = {
    'label': './data/strawberry/label.txt',
    'train': './data/strawberry/train/bbox.json',
    'valid': './data/strawberry/valid/bbox.json',
    'test': './data/strawberry/test/bbox.json',
    'samples': './data/strawberry/test/images',
    'test_result': './data/strawberry/test/test_outputs.bbox.json',
}
__segm_data = {
    'label': './data/strawberry/label.txt',
    'train': './data/strawberry/train/segm.json',
    'valid': './data/strawberry/valid/segm.json',
    'test': './data/strawberry/test/segm.json',
    'samples': './data/strawberry/test/images',
    'test_result': './data/strawberry/test/test_outputs.segm.json',
}
data = {
    'cls': __cls_data,
    'det': __det_data,
    'segm': __segm_data
}


def set_ws(dpath):
    dpath = os.path.join('outputs', dpath)
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    return dpath


def run_cmd(cmd):
    print('\nCOMMAND -----------------------------------------')
    print(' '.join(cmd))
    print('-------------------------------------------------\n')
    output = subprocess.run(cmd)
    if output.returncode != 0:
        raise Exception('Error: {}'.format(output.returncode))


class COCO():
    def __init__(self, file_path):
        with open(file_path) as infh:
            self.data = json.load(infh)

        self.images = set([_['file_name'] for _ in self.data['images']])
        self.annotations = set([_['id'] for _ in self.data['annotations']])
        self.categories = [_['name'] for _ in sorted(self.data['categories'], key=lambda x: x['id'])]

        
