import os
import numpy as np
import PIL.Image
import unittest
import testutils
import cvtk


class TestTorchScript(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def __run_proc(self, code_generator, vanilla):
        module = 'vanilla' if vanilla else 'cvtk'
        dpath = testutils.set_ws(f'torch_torch__{code_generator}_{module}')
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'api':
            cvtk.ml.deploy_model(script, backend='torch', task='cls', vanilla=vanilla)
        
        elif code_generator == 'script':
            cmd_ = ['cvtk', 'deploy-model', '--backend', 'torch', '--task', 'cls', '--script', script]
            if vanilla:
                cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)

        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data['cls']['label'],
                    '--train', testutils.data['cls']['train'],
                    '--valid', testutils.data['cls']['valid'],
                    '--test', testutils.data['cls']['test'],
                    '--output_weights', os.path.join(dpath, 'fruits.pth')])

        testutils.run_cmd(['python', script, 'test',
                    '--label', testutils.data['cls']['label'],
                    '--data', testutils.data['cls']['test'],
                    '--model_weights', os.path.join(dpath, 'fruits.pth'),
                    '--output', os.path.join(dpath, 'test_results.txt')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', testutils.data['cls']['label'],
                    '--data', testutils.data['cls']['samples'],
                    '--model_weights', os.path.join(dpath, 'fruits.pth'),
                    '--output', os.path.join(dpath, 'inference_results.txt')])


    def test_cvtk_api(self):
        self.__run_proc('api', False)


    def test_torch_api(self):
        self.__run_proc('api', True)


    def test_cvtk_script(self):
        self.__run_proc('script', False)


    def test_torch_script(self):
        self.__run_proc('script', True)    
    


class TestTorchUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('torch_torchutils')

        self.label = testutils.data['cls']['label']
        self.train = testutils.data['cls']['train']
        self.valid = testutils.data['cls']['valid']
        self.sample = testutils.data['cls']['samples']
        self.test = testutils.data['cls']['test']


    def test_square_resize_accepts_ndarray(self):
        image = np.zeros((20, 10, 3), dtype=np.uint8)
        image[:, :, 1] = 255
        resized = cvtk.ml.data.SquareResize(shape=32)(image)
        self.assertIsInstance(resized, PIL.Image.Image)
        self.assertEqual(resized.size, (32, 32))


    def test_iterable_dataset_len(self):
        datalabel = cvtk.ml.data.DataLabel(self.label)
        dataset = cvtk.ml.torchutils.Dataset(datalabel,
                                             self.train,
                                             transform=cvtk.ml.torchutils.DataTransform(224, is_train=False),
                                             stream_data=True)

        with open(self.train, 'r') as fh:
            expected = sum(1 for line in fh if line.rstrip().split('\t')[0:1] and line.rstrip().split('\t')[0] != '')

        self.assertEqual(len(dataset), expected)


    def __inference(self, model, datalabel, data, output_fpath):
        data = cvtk.ml.torchutils.DataLoader(
                cvtk.ml.torchutils.Dataset(datalabel,
                                           data,
                                           transform=cvtk.ml.torchutils.DataTransform(224, is_train=False)),
                batch_size=2,
                num_workers=8)
        probs = model.inference(data)
        probs.to_csv(output_fpath,
                     sep = '\t', header=True, index=True, index_label='image')



    def __test_torchutils(self, train, valid=None, test=None, output=None, batch_size=8, num_workers=8):
        temp_dpath = os.path.splitext(output)[0]

        datalabel = cvtk.ml.data.DataLabel(self.label)
        model = cvtk.ml.torchutils.ClsRunner(datalabel, 'resnet18', 'ResNet18_Weights.DEFAULT', temp_dpath)

        train = cvtk.ml.torchutils.DataLoader(
                cvtk.ml.torchutils.Dataset(datalabel,
                                           train,
                                           transform=cvtk.ml.torchutils.DataTransform(224, is_train=True)),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True)
        if valid is not None:
            valid = cvtk.ml.torchutils.DataLoader(
                        cvtk.ml.torchutils.Dataset(datalabel,
                                                   valid,
                                                   transform=cvtk.ml.torchutils.DataTransform(224, is_train=False)),
                        batch_size=batch_size,
                        num_workers=num_workers)
        if test is not None:
            test = cvtk.ml.torchutils.DataLoader(
                        cvtk.ml.torchutils.Dataset(datalabel,
                                                   test,
                                                   transform=cvtk.ml.torchutils.DataTransform(224, is_train=False)),
                        batch_size=batch_size,
                        num_workers=num_workers)

        model.train(train, valid, test, epoch=3)
        print('resume ...')
        model.train(train, valid, test, epoch=10, resume=True)
        model.save(output)

        cvtk.viz.plot(os.path.splitext(output)[0] + '.train_stats.txt',
                      x='epoch',
                      y=[['train_loss', 'valid_loss'], ['train_acc', 'valid_acc']],
                      output=os.path.splitext(output)[0] + '.train_stats.png')
        if test is not None:
            cvtk.viz.plot_cm(os.path.splitext(output)[0] + '.test_outputs.txt',
                    os.path.splitext(output)[0] + '.test_outputs.cm.png')
            

        model = cvtk.ml.torchutils.ClsRunner(datalabel, 'resnet18', output, temp_dpath)
        self.__inference(model, datalabel, self.sample, os.path.splitext(output)[0] + '.inference_results_dir.txt')
        self.__inference(model, datalabel, cvtk.io.imlist(self.sample), os.path.splitext(output)[0] + '.inference_results_list.txt')
        self.__inference(model, datalabel, cvtk.io.imlist(self.sample)[0], os.path.splitext(output)[0] + '.inference_results_single.txt')


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

