import os
import subprocess
import cvtk.ml.utils
import cvtk.ml.mmdet
import unittest


def make_dirs(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)


class TestMMDet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_mmdet')
        make_dirs(self.dpath)
    

    def __run_proc(self, task, module):
        dpath = os.path.join(self.dpath, f'{task}_{module}')
        make_dirs(dpath)
        djson = 'bbox.json' if task == 'det' else 'segm.json'
        script = os.path.join(dpath, 'script.py')

        cvtk.ml.utils.generate_source(script, task=task, module=module)

        cmd = ['python', script, 'train',
                    '--label', './data/strawberry/label.txt',
                    '--train', f'./data/strawberry/train/{djson}',
                    '--valid', f'./data/strawberry/valid/{djson}',
                    '--test', f'./data/strawberry/test/{djson}',
                    '--output_weights', os.path.join(dpath, 'sb.pth')]
        print(' '.join(cmd))
        output = subprocess.run(cmd)
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        cmd = ['python', script, 'inference',
                    '--label', './data/strawberry/label.txt',
                    '--data', './data/strawberry/test/images',
                    '--model_weights', os.path.join(dpath, 'sb.pth'),
                    '--output', os.path.join(dpath, 'pred_outputs')]
        print(' '.join(cmd))
        output = subprocess.run(cmd)
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
    

    def test_det_cvtk(self):
        self.__run_proc('det', 'cvtk')
        

    def test_det_mmdet(self):
        self.__run_proc('det', 'mmdet')


    def test_segm_cvtk(self):
        self.__run_proc('segm', 'cvtk')
        

    def test_segm_mmdet(self):
        self.__run_proc('segm', 'mmdet')






if __name__ == '__main__':
    unittest.main()
