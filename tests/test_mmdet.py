import os
import subprocess
import cvtk.ml.utils
import cvtk.ml.mmdet
import unittest
import testutils



class TestMMDet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_mmdet')
        testutils.make_dirs(self.dpath)
    

    def __run_proc(self, task, module, code_generator):
        dpath = os.path.join(self.dpath, f'{task}_{module}_{code_generator}')
        testutils.make_dirs(dpath)
        
        script = os.path.join(dpath, 'script.py')
        data = {
            'det': testutils.det_data,
            'segm': testutils.segm_data
        }
        
        if code_generator == 'source':
            cvtk.ml.utils.generate_source(script, task=task, module=module)
        elif code_generator == 'cmd':
            testutils.run_cmd(['cvtk', 'create',
                    '--task', task,
                    '--script', script,
                    '--module', module])

        testutils.run_cmd(['cvtk', 'create',
                    '--task', task,
                    '--script', script,
                    '--module', module])

        testutils.run_cmd(['python', script, 'train',
                    '--label', data[task]['label'],
                    '--train', data[task]['train'],
                    '--valid', data[task]['valid'],
                    '--test', data[task]['test'],
                    '--output_weights', os.path.join(dpath, 'sb.pth')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', data[task]['label'],
                    '--data', data[task]['samples'],
                    '--model_weights', os.path.join(dpath, 'sb.pth'),
                    '--output', os.path.join(dpath, 'pred_outputs')])
    

    def test_det_cvtk_cmd(self):
        self.__run_proc('det', 'cvtk', 'cmd')


    def test_det_cvtk_source(self):
        self.__run_proc('det', 'cvtk', 'source')


    def test_det_mmdet_cmd(self):
        self.__run_proc('det', 'mmdet', 'cmd')


    def test_det_mmdet_source(self):
        self.__run_proc('det', 'mmdet', 'source')


    def test_segm_cvtk_source(self):
        self.__run_proc('segm', 'cvtk', 'source')
        

    def test_segm_mmdet_cmd(self):
        self.__run_proc('segm', 'mmdet', 'cmd')





if __name__ == '__main__':
    unittest.main()
