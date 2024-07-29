import os
import cvtk.ml.utils
import unittest
import testutils



class TestMMDet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = testutils.set_ws(os.path.join('outputs', 'test_mmdet'))
    

    def __run_proc(self, task, vanilla, code_generator):
        module = 'vanilla' if vanilla else 'cvtk'
        dpath = testutils.set_ws(os.path.join(self.dpath, f'{task}_{module}_{code_generator}'))
        
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'source':
            cvtk.ml.utils.generate_source(script, task=task, vanilla=vanilla)
        elif code_generator == 'cmd':
            cmd_ = ['cvtk', 'create', '--task', task, '--script', script]
            if vanilla:
                cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)

        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data[task]['label'],
                    '--train', testutils.data[task]['train'],
                    '--valid', testutils.data[task]['valid'],
                    '--test', testutils.data[task]['test'],
                    '--output_weights', os.path.join(dpath, 'sb.pth')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', testutils.data[task]['label'],
                    '--data', testutils.data[task]['samples'],
                    '--model_weights', os.path.join(dpath, 'sb.pth'),
                    '--output', os.path.join(dpath, 'pred_outputs')])
    

    def test_det_cvtk_cmd(self):
        self.__run_proc('det', 'cvtk', 'cmd')


    def test_det_cvtk_source(self):
        self.__run_proc('det', 'cvtk', 'source')


    def test_det_mmdet_cmd(self):
        self.__run_proc('det', 'vanilla', 'cmd')


    def test_det_mmdet_source(self):
        self.__run_proc('det', 'vanilla', 'source')


    def test_segm_cvtk_source(self):
        self.__run_proc('segm', 'cvtk', 'source')
        

    def test_segm_mmdet_cmd(self):
        self.__run_proc('segm', 'vanilla', 'cmd')





if __name__ == '__main__':
    unittest.main()
