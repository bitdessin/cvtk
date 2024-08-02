import os
from cvtk import imlist, ImageDeck
from cvtk.ml import generate_source
from cvtk.ml.data import DataLabel
from cvtk.ml.mmdetutils import MMDETCORE, DataLabel, DataLoader, Dataset, DataPipeline, plot_trainlog
import unittest
import testutils



class TestScript(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('mmdet_script')
    

    def __run_proc(self, task, vanilla, code_generator):
        module = 'vanilla' if vanilla else 'cvtk'
        dpath = testutils.set_ws(f'mmdet_script__{task}_{module}_{code_generator}')
        
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'source':
            generate_source(script, task=task, vanilla=vanilla)
        elif code_generator == 'cmd':
            cmd_ = ['cvtk', 'create', '--task', task, '--script', script]
            if vanilla:
                cmd_.append('--vanilla')
            testutils.run_cmd(cmd_)

        testutils.run_cmd(['python', script, 'train',
                    '--label', testutils.data[task]['label'],
                    '--train', testutils.data[task]['train'],
                    '--valid', testutils.data[task]['valid'],
                    '--test', testutils.data[task]['test'],
                    '--output_weights', os.path.join(dpath, 'sb.pth')])

        testutils.run_cmd(['python', script, 'inference',
                    '--label', testutils.data[task]['label'],
                    '--data', testutils.data[task]['samples'],
                    '--model_weights', os.path.join(dpath, 'sb.pth'),
                    '--output', os.path.join(dpath, 'inference_results')])
    

    def test_det_cvtk_cmd(self):
        self.__run_proc('det', False, 'cmd')


    def test_det_cvtk_source(self):
        self.__run_proc('det', False, 'source')


    def test_det_mmdet_cmd(self):
        self.__run_proc('det', True, 'cmd')


    def test_det_mmdet_source(self):
        self.__run_proc('det', True, 'source')


    def test_segm_cvtk_source(self):
        self.__run_proc('segm', False, 'source')
        

    def test_segm_mmdet_cmd(self):
        self.__run_proc('segm', True, 'cmd')



class TestDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_dataset_none(self):
        dataset = Dataset(None, None, None)
        self.assertEqual(dataset, None)

    def test_dataset_list(self):
        imgs = ['img_1.jpg', 'img_2.jpg', 'img_3.jpg']
        dataset = Dataset(DataLabel(['tomato', 'eggplant', 'strawberry', 'cucumber']), imgs)
        self.assertEqual(dataset['data_root'], imgs)

    def test_dataset_str(self):
        image_dpath = 'data/fruits/images'
        dataset = Dataset(DataLabel(['leaf', 'root', 'flower']), image_dpath)
        self.assertEqual(dataset['data_root'],
                         os.path.join(os.path.dirname(__file__), image_dpath))
        
    def test_dataset_dict(self):
        labels = ['tomato', 'eggplant', 'strawberry', 'cucumber']
        data_dict = dict(
            metainfo=dict(classes=labels),
            type='CocoDataset',
            pipeline=None,
            data_root=''
        )
        dataset = Dataset(DataLabel(labels), data_dict)
        self.assertEqual(dataset, data_dict)




class TestDataLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def test_dataloader_none(self):
        self.assertRaises(ValueError,
                          DataLoader, None, phase='train')

        dataloader = DataLoader(None, phase='valid')
        self.assertEqual(dataloader,
                         dict(
                            val_dataloader=None,
                            val_cfg=None,
                            val_evaluator=None))

        dataloader = DataLoader(None, phase='test')
        self.assertEqual(dataloader,
                         dict(
                            test_dataloader=None,
                            test_cfg=None,
                            test_evaluator=None))

        dataloader = DataLoader(None, phase='inference')
        self.assertIsNotNone(dataloader)


        

class TestMMDet(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('mmdet_mmdet')
        self.sample = testutils.data['det']['samples']

    
    def __inference(self, model, datalabel, data, output_dpath):
        data = DataLoader(
                    Dataset(datalabel, data, DataPipeline()),
                    phase='inference', batch_size=2, num_workers=8)
        pred_outputs = model.inference(data)
        for im in pred_outputs:
            im.draw(format='bbox+segm',
                    output=output_dpath + os.path.basename(im.source))



    def __test_mmdetutils(self, label, train, valid=None, test=None, output_dpath=None, batch_size=2, num_workers=8):
        output_pfx = os.path.join(output_dpath, 'sb')
        datalabel = DataLabel(label)
        model = MMDETCORE(datalabel, "faster-rcnn_r101_fpn_1x_coco", None, workspace=output_dpath)

        train = DataLoader(
                    Dataset(datalabel, train,
                            DataPipeline(is_train=True, with_bbox=True, with_mask=False)),
                    phase='train', batch_size=batch_size, num_workers=num_workers)
        if valid is not None:
            valid = DataLoader(
                        Dataset(datalabel, valid,
                                DataPipeline(is_train=False, with_bbox=True, with_mask=False)),
                        phase='valid', batch_size=batch_size, num_workers=num_workers)
        if test is not None:
            test = DataLoader(
                        Dataset(datalabel, test,
                                DataPipeline(is_train=False, with_bbox=True, with_mask=False)),
                        phase='test', batch_size=batch_size, num_workers=num_workers)

        model.train(train, valid, test, epoch=5)
        model.save(f'{output_pfx}.pth')

        if os.path.exists(f'{output_pfx}.train_stats.train.txt'):
            plot_trainlog(f'{output_pfx}.train_stats.train.txt',
                          output=f'{output_pfx}.train_stats.train.png')
        if os.path.exists(f'{output_pfx}.train_stats.valid.txt'):
            plot_trainlog(f'{output_pfx}.train_stats.valid.txt',
                          output=f'{output_pfx}.train_stats.valid.png')


        # inference
        model = MMDETCORE(datalabel, f'{output_pfx}.py', f'{output_pfx}.pth',
                          workspace=output_dpath)
        
        #  images from a folder
        self.__inference(model, datalabel, self.sample, os.path.join(output_dpath, 'd_'))
        self.__inference(model, datalabel, imlist(self.sample), os.path.join(output_dpath, 'l_'))
        self.__inference(model, datalabel, imlist(self.sample)[0], os.path.join(output_dpath, 'f_'))



    def test_det_t_t_t(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            testutils.data['det']['valid'],
            testutils.data['det']['test'],
            os.path.join(self.ws, 'det_trainvalidtest'))


    def test_det_t_t_f(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            testutils.data['det']['valid'],
            None,
            os.path.join(self.ws, 'det_trainvalid'))


    def test_det_t_f_t(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            None,
            testutils.data['det']['test'],
            os.path.join(self.ws, 'det_traintest'))
    

    def test_det_t_f_f(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            None,
            None,
            os.path.join(self.ws, 'det_train'))


    def test_segm_t_t_t(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            testutils.data['segm']['valid'],
            testutils.data['segm']['test'],
            os.path.join(self.ws, 'segm_trainvalidtest'))


    def test_segm_t_t_f(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            testutils.data['segm']['valid'],
            None,
            os.path.join(self.ws, 'segm_trainvalid'))


    def test_segm_t_f_t(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            None,
            testutils.data['segm']['test'],
            os.path.join(self.ws, 'segm_traintest'))
    
    
    def test_segm_t_f_f(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            None,
            None,
            os.path.join(self.ws, 'segm_train'))



if __name__ == '__main__':
    unittest.main()

