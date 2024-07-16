import os
import subprocess
import cvtk.ml.torch
import unittest


def make_dirs(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTorch, self).__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_torch')
        make_dirs(self.dpath)
    

    def test_cls_cvtk(self):
        pfx = os.path.join(self.dpath, 'cvtkscript')

        cvtk.ml.torch.generate_source(f'{pfx}.py', module='cvtk')

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
                                 '--output', f'{pfx}.inference_results.txt'])
        if output.returncode != 0:
            raise Exception('Error: {}'.format(output.returncode))

        fig = cvtk.ml.torch.plot_trainlog(f'{pfx}.train_stats.txt',
                                          f'{pfx}.train_stats.png')
        fig.show()

        fig = cvtk.ml.torch.plot_cm(f'{pfx}.test_outputs.txt',
                                    f'{pfx}.test_outputs.png')
        fig.show()


    def test_cls_torch(self):
        pfx = os.path.join(self.dpath, 'torchscript')

        cvtk.ml.torch.generate_source(f'{pfx}.py', module='torch')

        subprocess.run(['python', f'{pfx}.py', 'train',
                        '--dataclass', './data/fruits/class.txt',
                        '--train', './data/fruits/train.txt',
                        '--valid', './data/fruits/valid.txt',
                        '--test', './data/fruits/test.txt',
                        '--output_weights', f'{pfx}.pth'])
        
        subprocess.run(['python', f'{pfx}.py', 'inference',
                        '--dataclass', './data/fruits/class.txt',
                        '--data', './data/fruits/test.txt',
                        '--model_weights', f'{pfx}.pth',
                        '--output', f'{pfx}.inference_results.txt'])

        fig = cvtk.ml.torch.plot_trainlog(f'{pfx}.train_stats.txt',
                                          f'{pfx}.train_stats.png')
        fig.show()

        fig = cvtk.ml.torch.plot_cm(f'{pfx}.test_outputs.txt',
                                    f'{pfx}.test_outputs.png')
        fig.show()      
        

if __name__ == '__main__':
    unittest.main()
