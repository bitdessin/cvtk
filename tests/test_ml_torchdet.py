import os
import unittest

import cvtk
import testutils


class TestTorchDet(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.ws = testutils.set_ws('torchdetapi_api')
		self.sample = testutils.data['det']['samples']
		self.sample_segm = testutils.data['segm']['samples']


	def __inference(self, model, datalabel, data, output_dpath, layers=None):
		if layers is None:
			layers = ['bbox']

		data = cvtk.ml.torchdetapi.DataLoader(
			cvtk.ml.torchdetapi.Dataset(
				datalabel,
				data,
				transform=cvtk.ml.torchdetapi.DataTransform(is_train=False),
			),
			batch_size=2,
			num_workers=0,
			shuffle=False,
		)
		pred_outputs = model.inference(data)

		self.assertIsInstance(pred_outputs, cvtk.data.ImageDataset)
		self.assertGreaterEqual(pred_outputs.size, 1)

		for im in pred_outputs:
			im.draw(layers=layers, output=output_dpath + os.path.basename(im.source))


	def __test_torchdetapi(self, label, train, valid=None, test=None, output_dpath=None, task='det', batch_size=1, num_workers=0, model_name=None):
		output_pfx = os.path.join(output_dpath, 'sb')
		datalabel = cvtk.ml.data.DataLabel(label)

		if task == 'segm':
			runner = cvtk.ml.torchdetapi.SegmRunner
			if model_name is None:
				model_name = 'maskrcnn_resnet50_fpn'
			sample_data = self.sample_segm
			draw_layers = ['bbox', 'segm']
		else:
			runner = cvtk.ml.torchdetapi.DetRunner
			if model_name is None:
				model_name = 'fasterrcnn_resnet50_fpn'
			sample_data = self.sample
			draw_layers = ['bbox']

		model = runner(
			datalabel,
			model_name,
			None,
			workspace=output_dpath
		)

		train_loader = cvtk.ml.torchdetapi.DataLoader(
			cvtk.ml.torchdetapi.Dataset(
				datalabel,
				train,
				transform=cvtk.ml.torchdetapi.DataTransform(is_train=True),
			),
			batch_size=batch_size,
			num_workers=num_workers,
			shuffle=True,
		)

		valid_loader = None
		if valid is not None:
			valid_loader = cvtk.ml.torchdetapi.DataLoader(
				cvtk.ml.torchdetapi.Dataset(
					datalabel,
					valid,
					transform=cvtk.ml.torchdetapi.DataTransform(is_train=False),
				),
				batch_size=batch_size,
				num_workers=num_workers,
				shuffle=False,
			)

		test_loader = None
		if test is not None:
			test_loader = cvtk.ml.torchdetapi.DataLoader(
				cvtk.ml.torchdetapi.Dataset(
					datalabel,
					test,
					transform=cvtk.ml.torchdetapi.DataTransform(is_train=False),
				),
				batch_size=batch_size,
				num_workers=num_workers,
				shuffle=False,
			)

		model.train(train_loader, valid_loader, test_loader, epoch=5)
		model.save(f'{output_pfx}.pth')

		self.assertTrue(os.path.exists(f'{output_pfx}.pth'))
		self.assertTrue(os.path.exists(f'{output_pfx}.dl.txt'))
		self.assertTrue(os.path.exists(f'{output_pfx}.train_stats.txt'))
		if test is not None:
			self.assertTrue(os.path.exists(f'{output_pfx}.test_stats.json'))
			self.assertIsNotNone(model.test_stats)

		# inference
		model = runner(
			datalabel,
			model_name,
			f'{output_pfx}.pth',
			workspace=output_dpath
		)

		self.__inference(model, datalabel, sample_data, os.path.join(output_dpath, 'd_'), layers=draw_layers)
		self.__inference(model, datalabel, cvtk.io.imlist(sample_data), os.path.join(output_dpath, 'l_'), layers=draw_layers)
		self.__inference(model, datalabel, cvtk.io.imlist(sample_data)[0], os.path.join(output_dpath, 'f_'), layers=draw_layers)


	def test_dataset_from_coco(self):
		datalabel = cvtk.ml.data.DataLabel(testutils.data['det']['label'])
		dataset = cvtk.ml.torchdetapi.Dataset(
			datalabel,
			testutils.data['det']['train'],
			transform=cvtk.ml.torchdetapi.DataTransform(is_train=True),
		)

		self.assertGreater(len(dataset), 0)
		self.assertIsNotNone(dataset.ann_file)

		img, target, path = dataset[0]
		self.assertTrue(hasattr(img, 'shape'))
		self.assertIsInstance(path, str)
		self.assertIsInstance(target, dict)
		self.assertIn('boxes', target)
		self.assertIn('labels', target)


	def test_dataset_from_coco_segm(self):
		datalabel = cvtk.ml.data.DataLabel(testutils.data['segm']['label'])
		dataset = cvtk.ml.torchdetapi.Dataset(
			datalabel,
			testutils.data['segm']['train'],
			transform=cvtk.ml.torchdetapi.DataTransform(is_train=True),
		)

		self.assertGreater(len(dataset), 0)
		self.assertIsNotNone(dataset.ann_file)

		img, target, path = dataset[0]
		self.assertTrue(hasattr(img, 'shape'))
		self.assertIsInstance(path, str)
		self.assertIsInstance(target, dict)
		self.assertIn('boxes', target)
		self.assertIn('labels', target)
		self.assertIn('masks', target)


	def test_det_t_t_t(self):
		self.__test_torchdetapi(
			testutils.data['det']['label'],
			testutils.data['det']['train'],
			testutils.data['det']['valid'],
			testutils.data['det']['test'],
			task='det',
			output_dpath=os.path.join(self.ws, 'det_trainvalidtest'),
		)


	def test_det_retinanet_t_t_t(self):
		self.__test_torchdetapi(
			testutils.data['det']['label'],
			testutils.data['det']['train'],
			testutils.data['det']['valid'],
			testutils.data['det']['test'],
			task='det',
			model_name='retinanet_resnet50_fpn',
			output_dpath=os.path.join(self.ws, 'det_retinanet_trainvalidtest'),
		)


	def test_det_ssd_t_t_t(self):
		self.__test_torchdetapi(
			testutils.data['det']['label'],
			testutils.data['det']['train'],
			testutils.data['det']['valid'],
			testutils.data['det']['test'],
			task='det',
			model_name='ssd300_vgg16',
			output_dpath=os.path.join(self.ws, 'det_ssd_trainvalidtest'),
		)


	def test_det_t_t_f(self):
		self.__test_torchdetapi(
			testutils.data['det']['label'],
			testutils.data['det']['train'],
			testutils.data['det']['valid'],
			None,
			task='det',
			output_dpath=os.path.join(self.ws, 'det_trainvalid'),
		)


	def test_det_t_f_t(self):
		self.__test_torchdetapi(
			testutils.data['det']['label'],
			testutils.data['det']['train'],
			None,
			testutils.data['det']['test'],
			task='det',
			output_dpath=os.path.join(self.ws, 'det_traintest'),
		)


	def test_det_t_f_f(self):
		self.__test_torchdetapi(
			testutils.data['det']['label'],
			testutils.data['det']['train'],
			None,
			None,
			task='det',
			output_dpath=os.path.join(self.ws, 'det_train'),
		)


	def test_segm_t_t_t(self):
		self.__test_torchdetapi(
			testutils.data['segm']['label'],
			testutils.data['segm']['train'],
			testutils.data['segm']['valid'],
			testutils.data['segm']['test'],
			task='segm',
			output_dpath=os.path.join(self.ws, 'segm_trainvalidtest'),
		)


	def test_segm_t_t_f(self):
		self.__test_torchdetapi(
			testutils.data['segm']['label'],
			testutils.data['segm']['train'],
			testutils.data['segm']['valid'],
			None,
			task='segm',
			output_dpath=os.path.join(self.ws, 'segm_trainvalid'),
		)


	def test_segm_t_f_t(self):
		self.__test_torchdetapi(
			testutils.data['segm']['label'],
			testutils.data['segm']['train'],
			None,
			testutils.data['segm']['test'],
			task='segm',
			output_dpath=os.path.join(self.ws, 'segm_traintest'),
		)


	def test_segm_t_f_f(self):
		self.__test_torchdetapi(
			testutils.data['segm']['label'],
			testutils.data['segm']['train'],
			None,
			None,
			task='segm',
			output_dpath=os.path.join(self.ws, 'segm_train'),
		)


if __name__ == '__main__':
	unittest.main()
