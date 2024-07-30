import os
import cvtk.ml
import unittest
import testutils


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = testutils.set_ws(os.path.join('outputs', 'test_torch'))
    

    def __run_proc(self, vanilla, code_generator):
        module = 'vanilla' if vanilla else 'cvtk'
        dpath = testutils.set_ws(os.path.join(self.dpath, f'{module}_{code_generator}'))
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'source':
            cvtk.ml.generate_source(script, task='cls', vanilla=vanilla)
        elif code_generator == 'cmd':
            cmd_ = ['cvtk', 'create', '--task', 'cls', '--script', script]
            if vanilla:
                cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)

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
        self.__run_proc(False, 'source')


    def test_torch_source(self):
        self.__run_proc(True, 'source')


    def test_cvtk_cmd(self):
        self.__run_proc(False, 'cmd')


    def test_torch_cmd(self):
        self.__run_proc(True, 'cmd')    
    

if __name__ == '__main__':
    unittest.main()
