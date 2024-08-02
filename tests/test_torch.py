import os
import unittest.util
from cvtk import imlist
from cvtk.ml import generate_source
from cvtk.ml.data import DataLabel
from cvtk.ml.torchutils import DataLabel, CLSCORE, DataLoader, Dataset, DataTransform, plot_trainlog, plot_cm
import unittest
import testutils


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def __run_proc(self, vanilla, code_generator):
        module = 'vanilla' if vanilla else 'cvtk'
        dpath = testutils.set_ws(f'torch_torch__{module}_{code_generator}')
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'source':
            generate_source(script, task='cls', vanilla=vanilla)
        elif code_generator == 'cmd':
            cmd_ = ['cvtk', 'create', '--task', 'cls', '--script', script]
            if vanilla:
                cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)

        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data['cls']['label'],
                    '--train', testutils.data['cls']['train'],
                    '--valid', testutils.data['cls']['valid'],
                    '--test', testutils.data['cls']['test'],
                    '--output_weights', os.path.join(dpath, 'fruits.pth')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', testutils.data['cls']['label'],
                    #'--data', TU.data['cls']['test'],
                    '--data', testutils.data['cls']['samples'],
                    '--model_weights', os.path.join(dpath, 'fruits.pth'),
                    '--output', os.path.join(dpath, 'inference_results.txt')])


    def test_cvtk_source(self):
        self.__run_proc(False, 'source')


    def test_torch_source(self):
        self.__run_proc(True, 'source')


    def test_cvtk_cmd(self):
        self.__run_proc(False, 'cmd')


    def test_torch_cmd(self):
        self.__run_proc(True, 'cmd')    
    


class TestTorchUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('torch_torchutils')

        self.label = testutils.data['cls']['label']
        self.train = testutils.data['cls']['train']
        self.valid = testutils.data['cls']['valid']
        self.sample = testutils.data['cls']['samples']
        self.test = testutils.data['cls']['test']


    def __inference(self, model, datalabel, data, output_fpath):
        data = DataLoader(
                Dataset(datalabel, data, transform=DataTransform(224, is_train=False)),
                batch_size=2, num_workers=8)
        probs = model.inference(data)
        probs.to_csv(output_fpath,
                     sep = '\t', header=True, index=True, index_label='image')



    def __test_torchutils(self, train, valid=None, test=None, output=None, batch_size=8, num_workers=8):
        temp_dpath = os.path.splitext(output)[0]

        datalabel = DataLabel(self.label)
        model = CLSCORE(datalabel, 'resnet18', 'ResNet18_Weights.DEFAULT', temp_dpath)

        train = DataLoader(
                Dataset(datalabel, train, transform=DataTransform(224, is_train=True)),
                batch_size=batch_size, num_workers=num_workers, shuffle=True)
        if valid is not None:
            valid = DataLoader(
                        Dataset(datalabel, valid, transform=DataTransform(224, is_train=False)),
                        batch_size=batch_size, num_workers=num_workers)
        if test is not None:
            test = DataLoader(
                        Dataset(datalabel, test, transform=DataTransform(224, is_train=False)),
                        batch_size=batch_size, num_workers=num_workers)

        model.train(train, valid, test, epoch=3)
        print('resume ...')
        model.train(train, valid, test, epoch=10, resume=True)
        model.save(output)

        plot_trainlog(os.path.splitext(output)[0] + '.train_stats.txt',
                      os.path.splitext(output)[0] + '.train_stats.png')
        if test is not None:
            plot_cm(os.path.splitext(output)[0] + '.test_outputs.txt',
                    os.path.splitext(output)[0] + '.test_outputs.cm.png')
            

        model = CLSCORE(datalabel, 'resnet18', output, temp_dpath)
        self.__inference(model, datalabel, self.sample, os.path.splitext(output)[0] + '.inference_results.txt')
        self.__inference(model, datalabel, imlist(self.sample), os.path.splitext(output)[0] + '.inference_results.txt')
        self.__inference(model, datalabel, imlist(self.sample)[0], os.path.splitext(output)[0] + '.inference_results.txt')


    def test_torchutils_t_f_f(self):
        self.__test_torchutils(self.train, None, None,
                               os.path.join(self.ws, 'train', 'fruits.pth'))

    def test_torchutils_t_t_f(self):
        self.__test_torchutils(self.train, self.valid, None,
                               os.path.join(self.ws, 'trainvalid', 'fruits.pth'))

    def test_torchutils_t_f_t(self):
        self.__test_torchutils(self.train, None, self.test,
                               os.path.join(self.ws, 'traintest', 'fruits.pth'))

    def test_torchutils_t_t_t(self):
        self.__test_torchutils(self.train, self.valid, self.test,
                               os.path.join(self.ws, 'trainvalidtest', 'fruits.pth'))        


if __name__ == '__main__':
    unittest.main()


