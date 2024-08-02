import os
from cvtk.ml import generate_source, generate_app
import unittest
import testutils



class TestDemoAPP(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
    
    def __run_proc(self, task, task_vanilla, api_vanilla, code_generator):
        task_module = 'vanilla' if task_vanilla else 'cvtk'
        api_module = 'vanilla' if api_vanilla else 'cvtk'
        dpath = testutils.set_ws(f'fastapi_demoapp__{task}_{task_module}_{api_module}_{code_generator}')
        
        script = os.path.join(dpath, 'script.py')
        model_weight = os.path.join(dpath, 'model.pth')
        model_cfg = 'resnet18' if task == 'cls' else os.path.splitext(model_weight)[0] + '.py'
        app_project = os.path.join(dpath, 'app')

        if code_generator == 'source':
            generate_source(script, task=task, vanilla=task_vanilla)
        elif code_generator == 'cmd':
            cmd_ = ['cvtk', 'create', '--task', task, '--script', script]
            cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)
        
        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data[task]['label'],
                    '--train', testutils.data[task]['train'],
                    '--valid', testutils.data[task]['valid'],
                    '--test', testutils.data[task]['test'],
                    '--output_weights', os.path.join(dpath, 'model.pth')])

       
        if code_generator == 'source':
            generate_app(app_project,
                         source=script,
                         label=testutils.data[task]['label'],
                         model=model_cfg,
                         weights=os.path.join(dpath, 'model.pth'),
                         vanilla=api_vanilla)
        elif code_generator == 'cmd':
            cmd_ = ['cvtk', 'app',
                    '--project', app_project, 
                    '--source', script,
                    '--label', testutils.data[task]['label'],
                    '--model', model_cfg,
                    '--weights', os.path.join(dpath, 'model.pth')]
            if api_vanilla:
                cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)
        
        #testutils.run_cmd(['uvicorn', app_project, '--host', '0.0.0.0', '--port', '8080', '--reload'])
        

    def test_cls(self):
        self.__run_proc('cls', True, True, 'source')
        self.__run_proc('cls', False, False, 'cmd')


    def test_det(self):
        self.__run_proc('det', True, True, 'source')
        self.__run_proc('det', False, False, 'cmd')

    
    def test_segm(self):
        self.__run_proc('segm', True, True, 'cmd')
        self.__run_proc('segm', False, False, 'source')


if __name__ == '__main__':
    unittest.main()
