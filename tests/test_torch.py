import os
import subprocess
import cvtk.ml.utils
import cvtk.ml.torch
import unittest
import testutils


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_torch')
        testutils.make_dirs(self.dpath)
    

    def __run_proc(self, module, code_generator):
        dpath = os.path.join(self.dpath, f'{module}_{code_generator}')
        testutils.make_dirs(dpath)
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'source':
            cvtk.ml.utils.generate_source(script, task='cls', module=module)
        elif code_generator == 'cmd':
            testutils.run_cmd(['cvtk', 'create',
                    '--task', 'cls',
                    '--script', script,
                    '--module', module])

        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.cls_data['label'],
                    '--train', testutils.cls_data['train'],
                    '--valid', testutils.cls_data['valid'],
                    '--test', testutils.cls_data['test'],
                    '--output_weights', os.path.join(dpath, 'fruits.pth')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', testutils.cls_data['label'],
                    #'--data', TU.cls_data['test'],
                    '--data', testutils.cls_data['samples'],
                    '--model_weights', os.path.join(dpath, 'fruits.pth'),
                    '--output', os.path.join(dpath, 'pred_outputs.txt')])


    def test_cvtk_source(self):
        self.__run_proc('cvtk', 'source')


    def test_torch_source(self):
        self.__run_proc('torch', 'source')


    def test_cvtk_cmd(self):
        self.__run_proc('cvtk', 'cmd')


    def test_torch_cmd(self):
        self.__run_proc('torch', 'cmd')    
    

if __name__ == '__main__':
    unittest.main()
