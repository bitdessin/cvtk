import os
import subprocess
import unittest


        
class TestScriptsBase(unittest.TestCase):
    def test_split_dataset(self):
        output = subprocess.run(['cvtk', 'split',
                                 '--input', './data/fruits/all.txt',
                                 '--output', './output/fruits_subset.txt',
                                 '--type', 'text',
                                 '--ratios', '6:3:1',
                                 '--shuffle', '--balanced'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))



class TestScriptsCLS(unittest.TestCase):
    def test_cls_cvtk(self):
        output = subprocess.run(['cvtk', 'create',
                                 '--project', 'cls.cvtk.py',
                                 '--type', 'cls',
                                 '--module', 'cvtk'])
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
        
        
    def test_cls_torch(self):
        output = subprocess.run(['cvtk', 'create',
                                 '--project', 'cls.torch.py',
                                 '--type', 'cls',
                                 '--module', 'torch'])
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





class TestScriptsCLSPipeline(unittest.TestCase):
    def test_pipeline(self):
        output = subprocess.run(['cvtk', 'split',
                                 '--input', './data/fruits/all.txt',
                                 '--output', './output/fruits_subset.txt',
                                 '--type', 'text',
                                 '--ratios', '6:3:1',
                                 '--shuffle', '--balanced'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['cvtk', 'create',
                                 '--project', 'cls.cvtk.py',
                                 '--type', 'cls',
                                 '--module', 'cvtk'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        output = subprocess.run(['python', 'cls.cvtk.py', 'train',
                        '--dataclass', './data/fruits/class.txt',
                        '--train', './output/fruits_subset.txt.1',
                        '--valid', './output/fruits_subset.txt.2',
                        '--test', './output/fruits_subset.txt.3',
                        '--output_weights', './output/fruits.pth'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        
        output = subprocess.run(['python', 'cls.cvtk.py', 'inference',
                        '--dataclass', './data/fruits/class.txt',
                        '--data', './output/fruits_subset.txt.3',
                        '--model_weights', './output/fruits.pth',
                        '--output', './output/fruits.inference_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))
        






if __name__ == '__main__':
    unittest.main()
