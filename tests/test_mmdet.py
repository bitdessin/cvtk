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
    

    def __run_proc(self, module):
        pfx = os.path.join(self.dpath, f'{module}script')

        cvtk.ml.utils.generate_source(f'{pfx}.py', task='det', module=module)

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/strawberry/class.txt',
                                 '--train', './data/strawberry/train/annotations.json',
                                 '--valid', './data/strawberry/valid/annotations.json',
                                 '--test', './data/strawberry/test/annotations.json',
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
        
        fig = cvtk.ml.mmdet.plot_trainlog(f'{pfx}_sb.train_stats.train.txt',
                                          output=f'{pfx}_sb.train_stats.train.png')
        fig.show()

        fig = cvtk.ml.mmdet.plot_trainlog(f'{pfx}_sb.train_stats.valid.txt',
                                          output=f'{pfx}_sb.train_stats.valid.png')
        fig.show()



    def test_det_cvtk(self):
        self.__run_proc('cvtk')
        

    def test_det_mmdet(self):
        self.__run_proc('mmdet')




if __name__ == '__main__':
    unittest.main()
