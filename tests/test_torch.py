import os
import subprocess
import cvtk.ml.torch
import unittest

class TestTorch(unittest.TestCase):

    def cls_cvtk(self):
        cvtk.ml.torch.generate_source('cls.cvtk.py', module='cvtk')

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

        fig = cvtk.ml.torch.plot_trainlog('output/fruits.train_stats.txt', 'output/fruits.train_stats.png')
        fig.show()

        fig = cvtk.ml.torch.plot_cm('output/fruits.test_outputs.txt', 'output/fruits.cm.png')
        fig.show()


    def cls_torch(self):
        cvtk.ml.torch.generate_source('cls.torch.py', module='torch')

        subprocess.run(['python', 'cls.cvtk.py', 'train',
                        '--dataclass', './data/fruits/class.txt',
                        '--train', './data/fruits/train.txt',
                        '--valid', './data/fruits/valid.txt',
                        '--test', './data/fruits/test.txt',
                        '--output_weights', './output/fruits.pth'])
        
        subprocess.run(['python', 'cls.cvtk.py', 'inference',
                        '--dataclass', './data/fruits/class.txt',
                        '--data', './data/fruits/test.txt',
                        '--model_weights', './output/fruits.pth',
                        '--output', './output/fruits.inference_results.txt'])

        fig = cvtk.ml.torch.plot_trainlog('output/fruits.train_stats.txt', 'output/fruits.train_stats.png')
        fig.show()

        fig = cvtk.ml.torch.plot_cm('output/fruits.test_outputs.txt', 'output/fruits.cm.png')
        fig.show()      
        

if __name__ == '__main__':
    unittest.main()
