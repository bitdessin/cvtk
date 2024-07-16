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
        

    def test_cls_cvtk(self):
        pfx = os.path.join(self.dpath, 'cvtkscript')

        output = subprocess.run(['cvtk', 'create',
                                 '--project', f'{pfx}.py',
                                 '--type', 'cls',
                                 '--module', 'cvtk'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--train', './data/fruits/train.txt',
                                 '--valid', './data/fruits/valid.txt',
                                 '--test', './data/fruits/test.txt',
                                 '--output_weights', f'{pfx}.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python', f'{pfx}.py', 'inference',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--data', './data/fruits/test.txt',
                                 '--model_weights', f'{pfx}.pth',
                                 '--output', f'{pfx}.infrence_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
    
    
    def test_cls_torch(self):
        pfx = os.path.join(self.dpath, 'torchscript')

        output = subprocess.run(['cvtk', 'create',
                                 '--project', f'{pfx}.py',
                                 '--type', 'cls',
                                 '--module', 'torch'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--train', './data/fruits/train.txt',
                                 '--valid', './data/fruits/valid.txt',
                                 '--test', './data/fruits/test.txt',
                                 '--output_weights', f'{pfx}.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python', f'{pfx}.py', 'inference',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--data', './data/fruits/test.txt',
                                 '--model_weights', f'{pfx}.pth',
                                 '--output', f'{pfx}.infrence_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))



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
                                 '--project', f'{pfx}.py',
                                 '--type', 'cls',
                                 '--module', 'cvtk'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', f'{pfx}.py', 'train',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--train', f'{pfx}.txt.0',
                                 '--valid', f'{pfx}.txt.1',
                                 '--test', f'{pfx}.txt.2',
                                 '--output_weights', f'{pfx}.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python',f'{pfx}.py', 'inference',
                                 '--dataclass', './data/fruits/class.txt',
                                 '--data', f'{pfx}.txt.2',
                                 '--model_weights', f'{pfx}.pth',
                                 '--output', f'{pfx}.inference_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        



if __name__ == '__main__':
    unittest.main()
