import os
import subprocess
import cvtk.ml.utils
import cvtk.ml.torch
import unittest


def make_dirs(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_torch')
        make_dirs(self.dpath)
    

    def __run_proc(self, module):
        dpath = os.path.join(self.dpath, module)
        make_dirs(dpath)
        script = os.path.join(dpath, 'script.py')
        
        cvtk.ml.utils.generate_source(script, task='cls', module=module)

        output = subprocess.run(['python', script, 'train',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--train', './data/fruits/train.txt',
                                 '--valid', './data/fruits/valid.txt',
                                 '--test', './data/fruits/test.txt',
                                 '--output_weights', os.path.join(dpath, 'fruits.pth')])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', script, 'inference',
                                 '--dataclass', './data/fruits/class.txt',
                                 #'--data', './data/fruits/test.txt',
                                 '--data', './data/fruits/images',
                                 '--model_weights', os.path.join(dpath, 'fruits.pth'),
                                 '--output', os.path.join(dpath, 'pred_outputs.txt')])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))


    def test_cls_cvtk(self):
        self.__run_proc('cvtk')


    def test_cls_torch(self):
        self.__run_proc('torch')
        

if __name__ == '__main__':
    unittest.main()
