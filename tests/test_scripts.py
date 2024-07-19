import os
import subprocess
import unittest


def make_dirs(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)



class TestScriptsBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_scripts_base')
        make_dirs(self.dpath)


    def test_split_dataset(self):
        output = subprocess.run(['cvtk', 'split',
                                 '--input', './data/fruits/all.txt',
                                 '--output', os.path.join(self.dpath, 'fruits_subset.txt'),
                                 '--type', 'text',
                                 '--ratios', '6:3:1',
                                 '--shuffle', '--balanced'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))



class TestScriptsCLS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_scripts_cls')
        make_dirs(self.dpath)
    

    def __run_proc(self, module):
        pfx = os.path.join(self.dpath, f'{module}script')

        output = subprocess.run(['cvtk', 'create',
                                 '--script', f'{pfx}.py',
                                 '--task', 'cls',
                                 '--module', module])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--train', './data/fruits/train.txt',
                                 '--valid', './data/fruits/valid.txt',
                                 '--test', './data/fruits/test.txt',
                                 '--output_weights', f'{pfx}_fruits.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python', f'{pfx}.py', 'inference',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--data', './data/fruits/images',
                                 '--model_weights', f'{pfx}_fruits.pth',
                                 '--output', f'{pfx}_pred_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        

    def test_cls_cvtk(self):
        self.__run_proc('cvtk')
    
    
    def test_cls_torch(self):
        self.__run_proc('torch')



class TestScriptsCLSPipeline(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_scripts_cls_pipeline')
        make_dirs(self.dpath)


    def test_pipeline(self):
        pfx = os.path.join(self.dpath, 'pipeline')
        
        output = subprocess.run(['cvtk', 'split',
                                 '--input', './data/fruits/all.txt',
                                 '--output', f'{pfx}.txt',
                                 '--type', 'text',
                                 '--ratios', '6:3:1',
                                 '--shuffle', '--balanced'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['cvtk', 'create',
                                 '--script', f'{pfx}.py',
                                 '--task', 'cls',
                                 '--module', 'cvtk'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--train', f'{pfx}.txt.0',
                                 '--valid', f'{pfx}.txt.1',
                                 '--test', f'{pfx}.txt.2',
                                 '--output_weights', f'{pfx}_fruits.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python',f'{pfx}.py', 'inference',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--data', './data/fruits/images',
                                 '--model_weights', f'{pfx}_fruits.pth',
                                 '--output', f'{pfx}_pred_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        

class TestScriptsDET(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_scripts_det')
        make_dirs(self.dpath)
    

    def __run_proc(self, module):
        pfx = os.path.join(self.dpath, f'{module}script')

        output = subprocess.run(['cvtk', 'create',
                                 '--script', f'{pfx}.py',
                                 '--task', 'det',
                                 '--module', module])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/strawberry/class.txt',
                                 '--train', './data/strawberry/train/bbox.json',
                                 '--valid', './data/strawberry/valid/bbox.json',
                                 '--test', './data/strawberry/test/bbox.json',
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
        self.__run_proc('cvtk')
    
    
    def test_det_mmdet(self):
        self.__run_proc('torch')


class TestScriptsSEGM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_scripts_det')
        make_dirs(self.dpath)
    

    def __run_proc(self, module):
        pfx = os.path.join(self.dpath, f'{module}script')

        output = subprocess.run(['cvtk', 'create',
                                 '--script', f'{pfx}.py',
                                 '--task', 'segm',
                                 '--module', module])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/strawberry/class.txt',
                                 '--train', './data/strawberry/train/segm.json',
                                 '--valid', './data/strawberry/valid/segm.json',
                                 '--test', './data/strawberry/test/segm.json',
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
        

    def test_segm_cvtk(self):
        self.__run_proc('cvtk')
    
    
    def test_segm_mmdet(self):
        self.__run_proc('torch')


if __name__ == '__main__':
    unittest.main()
