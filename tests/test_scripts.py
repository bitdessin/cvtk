import os
import subprocess
import unittest


class TestScripts(unittest.TestCase):
    def cls_cvtk(self):
        output = subprocess.run(['cvtk', 'create', '--project', 'cls.cvtk.py', '--type', 'cls', '--module', 'cvtk'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', 'cls.cvtk.py', 'train',
                        '--dataclass', './data/fruits/class.txt',
                        '--train', './data/fruits/train.txt',
                        '--valid', './data/fruits/valid.txt',
                        '--test', './data/fruits/test.txt',
                        '--output_weights', './output/fruits.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python', 'cls.cvtk.py', 'inference',
                        '--dataclass', './data/fruits/class.txt',
                        '--data', './data/fruits/test.txt',
                        '--model_weights', './output/fruits.pth',
                        '--output', './output/fruits.inference_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        
    def cls_torch(self):
        output = subprocess.run(['cvtk', 'create', '--project', 'cls.torch.py', '--type', 'cls', '--module', 'torch'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', 'cls.cvtk.py', 'train',
                        '--dataclass', './data/fruits/class.txt',
                        '--train', './data/fruits/train.txt',
                        '--valid', './data/fruits/valid.txt',
                        '--test', './data/fruits/test.txt',
                        '--output_weights', './output/fruits.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python', 'cls.cvtk.py', 'inference',
                        '--dataclass', './data/fruits/class.txt',
                        '--data', './data/fruits/test.txt',
                        '--model_weights', './output/fruits.pth',
                        '--output', './output/fruits.inference_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        

if __name__ == '__main__':
    unittest.main()
