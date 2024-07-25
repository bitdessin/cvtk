import os
import subprocess
import cvtk.ml.utils
import cvtk.ml.mmdet
import cvtk.ml.torch
import unittest
import testutils




class TestFastapi(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = testutils.set_ws(os.path.join('outputs', 'test_fastapi'))
    
    
    def __run_proc(self, task, module, code_generator):
        dpath = testutils.set_ws(os.path.join(self.dpath, f'{task}_{module}_{code_generator}'))
        
        script = os.path.join(dpath, 'script.py')
        model_weight = os.path.join(dpath, 'model.pth')
        model_cfg = 'resnet18' if task == 'cls' else os.path.splitext(model_weight)[0] + '.py'
        app_project = os.path.join(dpath, 'app')

        if code_generator == 'source':
            cvtk.ml.utils.generate_source(script, task=task, module=module)
        elif code_generator == 'cmd':
            testutils.run_cmd(['cvtk', 'create',
                    '--task', task,
                    '--script', script,
                    '--module', module])
        
        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data[task]['label'],
                    '--train', testutils.data[task]['train'],
                    '--valid', testutils.data[task]['valid'],
                    '--test', testutils.data[task]['test'],
                    '--output_weights', os.path.join(dpath, 'model.pth')])

       
        if code_generator == 'source':
            cvtk.ml.utils.generate_app(app_project,
                            source=script,
                            label=testutils.data[task]['label'],
                            model=model_cfg,
                            weights=os.path.join(dpath, 'model.pth'),
                            module=module)
        elif code_generator == 'cmd':
            testutils.run_cmd(['cvtk', 'app',
                    '--project', app_project, 
                    '--source', script,
                    '--label', testutils.data[task]['label'],
                    '--model', model_cfg,
                    '--weights', os.path.join(dpath, 'model.pth'),
                    '--module', module])
        
        #testutils.run_cmd(['uvicorn', app_project, '--host', '0.0.0.0', '--port', '8080', '--reload'])
        

    def test_cls_cvtk_source(self):
        self.__run_proc('cls', 'cvtk', 'source')

    
    def test_cls_cvtk_cmd(self):
        self.__run_proc('cls', 'cvtk', 'cmd')


    def test_cls_torch_source(self):
        self.__run_proc('cls', 'torch', 'source')


    def test_cls_torch_cmd(self):
        self.__run_proc('cls', 'torch', 'cmd')


    def test_det_cvtk_source(self):
#        det_cvtk_source
        self.__run_proc('det', 'cvtk', 'source')

    
    def test_det_cvtk_cmd(self):
        ###
        self.__run_proc('det', 'cvtk', 'cmd')
        

    def test_segm_mmdet_source(self):
        self.__run_proc('segm', 'mmdet', 'source')


    def test_segm_mmdet_cmd(self):
        self.__run_proc('segm', 'mmdet', 'cmd')



if __name__ == '__main__':
    unittest.main()
