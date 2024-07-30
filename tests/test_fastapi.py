import os
import subprocess
import cvtk.ml
import unittest
import testutils




class TestFastapi(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = testutils.set_ws(os.path.join('outputs', 'test_fastapi'))
        self.module_types = [True, False]
        self.generator_types = ['source', 'cmd']
    
    
    def __run_proc(self, task, task_vanilla, api_vanilla, code_generator):
        task_module = 'vanilla' if task_vanilla else 'cvtk'
        api_module = 'vanilla' if api_vanilla else 'cvtk'
        dpath = testutils.set_ws(os.path.join(self.dpath, f'{task}_{task_module}_{api_module}_{code_generator}'))
        
        script = os.path.join(dpath, 'script.py')
        model_weight = os.path.join(dpath, 'model.pth')
        model_cfg = 'resnet18' if task == 'cls' else os.path.splitext(model_weight)[0] + '.py'
        app_project = os.path.join(dpath, 'app')

        if code_generator == 'source':
            cvtk.ml.generate_source(script, task=task, vanilla=task_vanilla)
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
            cvtk.ml.generate_app(app_project,
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
        for task_module in self.module_types:
            for api_module in self.module_types:
                for code_generator in self.generator_types:
                    self.__run_proc('cls', task_module, api_module, code_generator)


    def test_det(self):
        for task_module in self.module_types:
            for api_module in self.module_types:
                for code_generator in self.generator_types:
                    self.__run_proc('det', task_module, api_module, code_generator)

    
    def test_segm(self):
        for task_module in self.module_types:
            for api_module in self.module_types:
                for code_generator in self.generator_types:
                    self.__run_proc('segm', task_module, api_module, code_generator)


if __name__ == '__main__':
    unittest.main()
