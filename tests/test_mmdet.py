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
        pfx = os.path.join(self.dpath, f'{module}{task}script')
        djson = 'bbox.json' if task == 'det' else 'segm.json'

        cvtk.ml.utils.generate_source(f'{pfx}.py', task=task, module=module)

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/strawberry/class.txt',
                                 '--train', f'./data/strawberry/train/{djson}',
                                 '--valid', f'./data/strawberry/valid/{djson}',
                                 '--test', f'./data/strawberry/test/{djson}',
                                 '--output_weights', f'{pfx}_sb.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'inference',
                                 '--dataclass', './data/strawberry/class.txt',
                                 '--data', './data/strawberry/test/images',
                                 '--model_weights', f'{pfx}_sb.pth',
                                 '--output', f'{pfx}_pred_results.txt'])
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
