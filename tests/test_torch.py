import os
import cvtk.ml.utils
import unittest
import testutils


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = testutils.set_ws(os.path.join('outputs', 'test_torch'))
    

    def __run_proc(self, module, code_generator):
        dpath = testutils.set_ws(os.path.join(self.dpath, f'{module}_{code_generator}'))
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'source':
            cvtk.ml.utils.generate_source(script, task='cls', module=module)
        elif code_generator == 'cmd':
            testutils.run_cmd(['cvtk', 'create',
                    '--task', 'cls',
                    '--script', script,
                    '--module', module])

        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data['cls']['label'],
                    '--train', testutils.data['cls']['train'],
                    '--valid', testutils.data['cls']['valid'],
                    '--test', testutils.data['cls']['test'],
                    '--output_weights', os.path.join(dpath, 'fruits.pth')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', testutils.data['cls']['label'],
                    #'--data', TU.data['cls']['test'],
                    '--data', testutils.data['cls']['samples'],
                    '--model_weights', os.path.join(dpath, 'fruits.pth'),
                    '--output', os.path.join(dpath, 'pred_outputs.txt')])


    def test_cvtk_source(self):
        self.__run_proc('cvtk', 'source')


    def test_torch_source(self):
        self.__run_proc('vanilla', 'source')


    def test_cvtk_cmd(self):
        self.__run_proc('cvtk', 'cmd')


    def test_torch_cmd(self):
        self.__run_proc('vanilla', 'cmd')    
    

if __name__ == '__main__':
    unittest.main()
