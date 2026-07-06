import os
import cvtk
import unittest
import testutils


class TestScript(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('mmdet_script')
    

    def __run_proc(self, code_generator, vanilla, task):
        module = 'vanilla' if vanilla else 'cvtk'
        dpath = testutils.set_ws(os.path.join(self.ws, f'{module}_{code_generator}_{task}'))
        
        script = os.path.join(dpath, 'script.py')
        
        if code_generator == 'api':
            cvtk.ml.deploy_model(script, backend='mmdet', task=task, vanilla=vanilla)
        elif code_generator == 'script':
            cmd_ = ['cvtk', 'deploy-model', '--backend', 'mmdet', '--task', task, '--script', script]
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
        self.__run_proc('script', False, 'det')


    def test_det_cvtk_source(self):
        self.__run_proc('api', False, 'det')


    def test_det_mmdet_cmd(self):
        self.__run_proc('script', True, 'det')


    def test_det_mmdet_source(self):
        self.__run_proc('api', True, 'det')


    def test_segm_cvtk_source(self):
        self.__run_proc('api', False, 'segm')
        

    def test_segm_mmdet_cmd(self):
        self.__run_proc('script', True, 'segm')


class TestMMDet(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('mmdet_api')
        self.sample = testutils.data['det']['samples']

    
    def __inference(self, model, datalabel, data, output_dpath):
        data = cvtk.ml.mmdetutils.DataLoader(
                    cvtk.ml.mmdetutils.Dataset(datalabel, data, 
                                               cvtk.ml.mmdetutils.DataPipeline()),
                    phase='inference', batch_size=4, num_workers=8)
        pred_outputs = model.inference(data)
        for im in pred_outputs:
            im.draw(layers=['bbox', 'segm'],
                    output=output_dpath + os.path.basename(im.source))


    def __test_mmdetutils(self, label, train, valid=None, test=None, output_dpath=None, task='det', batch_size=4, num_workers=8):
        output_pfx = os.path.join(output_dpath, 'sb')
        datalabel = cvtk.ml.data.DataLabel(label)
        if task == 'det':
            model = cvtk.ml.mmdetutils.DetRunner(
                datalabel, "faster-rcnn_r101_fpn_1x_coco", None, workspace=output_dpath)
            runner_cls = cvtk.ml.mmdetutils.DetRunner
        else:
            model = cvtk.ml.mmdetutils.SegmRunner(
                datalabel, "mask-rcnn_r101_fpn_1x_coco", None, workspace=output_dpath)
            runner_cls = cvtk.ml.mmdetutils.SegmRunner

        with_mask = False if task == 'det' else True
        train = cvtk.ml.mmdetutils.DataLoader(
                    cvtk.ml.mmdetutils.Dataset(datalabel, train,
                                               cvtk.ml.mmdetutils.DataPipeline(is_train=True, with_bbox=True, with_mask=with_mask)),
                    phase='train', batch_size=batch_size, num_workers=num_workers)
        if valid is not None:
            valid = cvtk.ml.mmdetutils.DataLoader(
                        cvtk.ml.mmdetutils.Dataset(datalabel, valid,
                                                   cvtk.ml.mmdetutils.DataPipeline(is_train=False, with_bbox=True, with_mask=with_mask)),
                        phase='valid', batch_size=batch_size, num_workers=num_workers)
        if test is not None:
            test = cvtk.ml.mmdetutils.DataLoader(
                        cvtk.ml.mmdetutils.Dataset(datalabel, test,
                                                   cvtk.ml.mmdetutils.DataPipeline(is_train=False, with_bbox=True, with_mask=with_mask)),
                        phase='test', batch_size=batch_size, num_workers=num_workers)

        model.train(train, valid, test, epoch=10)
        model.save(f'{output_pfx}.pth')

        if os.path.exists(f'{output_pfx}.train_stats.train.txt'):
            cvtk.viz.plot(f'{output_pfx}.train_stats.train.txt',
                          x='epoch',
                          y=['loss', 'loss_cls', 'loss_bbox', 'acc'],
                          output=f'{output_pfx}.train_stats.train.png')
        if os.path.exists(f'{output_pfx}.train_stats.valid.txt'):
            cvtk.viz.plot(f'{output_pfx}.train_stats.valid.txt',
                          x='step',
                          y=['coco/bbox_mAP', 'coco/bbox_mAP_50'],
                          output=f'{output_pfx}.train_stats.valid.png')

        # inference
        model = runner_cls(datalabel, f'{output_pfx}.py', f'{output_pfx}.pth',
                          workspace=output_dpath)
        
        #  images from a folder
        self.__inference(model, datalabel, self.sample, os.path.join(output_dpath, 'd_'))
        self.__inference(model, datalabel, cvtk.io.imlist(self.sample), os.path.join(output_dpath, 'l_'))
        self.__inference(model, datalabel, cvtk.io.imlist(self.sample)[0], os.path.join(output_dpath, 'f_'))


    def test_det_t_t_t(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            testutils.data['det']['valid'],
            testutils.data['det']['test'],
            os.path.join(self.ws, 'det_trainvalidtest'),
            'det')


    def test_det_t_t_f(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            testutils.data['det']['valid'],
            None,
            os.path.join(self.ws, 'det_trainvalid'),
            'det')


    def test_det_t_f_t(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            None,
            testutils.data['det']['test'],
            os.path.join(self.ws, 'det_traintest'),
            'det')
    

    def test_det_t_f_f(self):
        self.__test_mmdetutils(
            testutils.data['det']['label'],
            testutils.data['det']['train'],
            None,
            None,
            os.path.join(self.ws, 'det_train'),
            'det')


    def test_segm_t_t_t(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            testutils.data['segm']['valid'],
            testutils.data['segm']['test'],
            os.path.join(self.ws, 'segm_trainvalidtest'),
            'segm')


    def test_segm_t_t_f(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            testutils.data['segm']['valid'],
            None,
            os.path.join(self.ws, 'segm_trainvalid'),
            'segm')


    def test_segm_t_f_t(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            None,
            testutils.data['segm']['test'],
            os.path.join(self.ws, 'segm_traintest'),
            'segm')
    
    
    def test_segm_t_f_f(self):
        self.__test_mmdetutils(
            testutils.data['segm']['label'],
            testutils.data['segm']['train'],
            None,
            None,
            os.path.join(self.ws, 'segm_train'),
            'segm')


if __name__ == '__main__':
    unittest.main()
