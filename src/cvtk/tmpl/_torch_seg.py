import os
import random
import cvtk


def train(
	label,
	train,
	valid,
	test,
	output_weights,
	input_weights=None,
	model_name='maskrcnn_resnet50_fpn',
	batch_size=2,
	num_workers=0,
	epoch=10,
	device='auto'
):
	temp_dpath = os.path.splitext(output_weights)[0]

	datalabel = cvtk.ml.data.DataLabel(label)
	model = cvtk.ml.torchdet.SegmRunner(
		datalabel,
		model=model_name,
		weights=input_weights,
		workspace=temp_dpath,
		device=device,
	)

	train_loader = cvtk.ml.torchdet.DataLoader(
		cvtk.ml.torchdet.Dataset(
			datalabel,
			train,
			transform=cvtk.ml.torchdet.DataTransform(is_train=True),
		),
		batch_size=batch_size,
		num_workers=num_workers,
		shuffle=True,
	)

	valid_loader = None
	if valid is not None:
		valid_loader = cvtk.ml.torchdet.DataLoader(
			cvtk.ml.torchdet.Dataset(
				datalabel,
				valid,
				transform=cvtk.ml.torchdet.DataTransform(is_train=False),
			),
			batch_size=batch_size,
			num_workers=num_workers,
			shuffle=False,
		)

	test_loader = None
	if test is not None:
		test_loader = cvtk.ml.torchdet.DataLoader(
			cvtk.ml.torchdet.Dataset(
				datalabel,
				test,
				transform=cvtk.ml.torchdet.DataTransform(is_train=False),
			),
			batch_size=batch_size,
			num_workers=num_workers,
			shuffle=False,
		)

	model.train(train_loader, valid_loader, test_loader, epoch=epoch)
	model.save(output_weights)


def inference(
	label,
	data,
	model_weights,
	output,
	model_name='maskrcnn_resnet50_fpn',
	cutoff=0.5,
	batch_size=2,
	num_workers=0,
	device='auto'
):
	datalabel = cvtk.ml.data.DataLabel(label)
	model = cvtk.ml.torchdet.SegmRunner(
		datalabel,
		model=model_name,
		weights=model_weights,
		workspace=output,
		device=device,
	)

	data_loader = cvtk.ml.torchdet.DataLoader(
		cvtk.ml.torchdet.Dataset(
			datalabel,
			data,
			transform=cvtk.ml.torchdet.DataTransform(is_train=False),
		),
		batch_size=batch_size,
		num_workers=num_workers,
		shuffle=False,
	)

	pred_outputs = model.inference(data_loader, cutoff=cutoff)

	if output is not None and not os.path.exists(output):
		os.makedirs(output)

	for im in pred_outputs:
		random.seed(1)
		if output is not None:
			im.draw(layers=['bbox', 'segm'], output=os.path.join(output, os.path.basename(im.source)))

	cocodict = pred_outputs.to_coco()
	if output is not None:
		cvtk.utils.save_json(cocodict, os.path.join(output, 'instances.coco.json'))


def _train(args):
	train(
		args.label,
		args.train,
		args.valid,
		args.test,
		args.output_weights,
		args.input_weights,
		args.model_name,
		args.batch_size,
		args.num_workers,
		args.epoch,
		args.device,
	)


def _inference(args):
	inference(
		args.label,
		args.data,
		args.model_weights,
		args.output,
		args.model_name,
		args.cutoff,
		args.batch_size,
		args.num_workers,
		args.device,
	)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--label', type=str, required=True)
	parser_train.add_argument('--train', type=str, required=True)
	parser_train.add_argument('--valid', type=str, required=False, default=None)
	parser_train.add_argument('--test', type=str, required=False, default=None)
	parser_train.add_argument('--input_weights', type=str, required=False, default=None)
	parser_train.add_argument('--output_weights', type=str, required=True)
	parser_train.add_argument('--model_name', type=str, default='maskrcnn_resnet50_fpn')
	parser_train.add_argument('--batch_size', type=int, default=2)
	parser_train.add_argument('--num_workers', type=int, default=0)
	parser_train.add_argument('--epoch', type=int, default=10)
	parser_train.add_argument('--device', type=str, default='auto')
	parser_train.set_defaults(func=_train)

	parser_inference = subparsers.add_parser('inference')
	parser_inference.add_argument('--label', type=str, required=True)
	parser_inference.add_argument('--data', type=str, required=True)
	parser_inference.add_argument('--model_weights', type=str, required=True)
	parser_inference.add_argument('--output', type=str, required=False, default='.')
	parser_inference.add_argument('--model_name', type=str, default='maskrcnn_resnet50_fpn')
	parser_inference.add_argument('--cutoff', type=float, default=0.5)
	parser_inference.add_argument('--batch_size', type=int, default=2)
	parser_inference.add_argument('--num_workers', type=int, default=0)
	parser_inference.add_argument('--device', type=str, default='auto')
	parser_inference.set_defaults(func=_inference)

	args = parser.parse_args()
	args.func(args)


"""
Example Usage:


python __SCRIPTNAME__ train \
	--label ./data/strawberry/label.txt \
	--train ./data/strawberry/train/segm.json \
	--valid ./data/strawberry/valid/segm.json \
	--test ./data/strawberry/test/segm.json \
	--output_weights ./output/sb.pth


python __SCRIPTNAME__ inference \
	--label ./data/strawberry/label.txt \
	--data ./data/strawberry/test/images \
	--model_weights ./output/sb.pth \
	--output ./output/pred_results

"""
