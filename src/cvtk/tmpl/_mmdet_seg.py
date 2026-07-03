import os
import random
import pandas as pd
import cvtk


def train(
    label,
    train,
    valid,
    test,
    output_weights,
    batch_size=4,
    num_workers=8,
    epoch=10
):
    temp_dpath = os.path.splitext(output_weights)[0]

    datalabel = cvtk.ml.data.DataLabel(label)
    model = cvtk.ml.mmdetutils.DetRunner(
        datalabel, "mask-rcnn_r101_fpn_1x_coco", None, workspace=temp_dpath)

    train = cvtk.ml.mmdetutils.DataLoader(
                cvtk.ml.mmdetutils.Dataset(datalabel, train,
                        cvtk.ml.mmdetutils.DataPipeline(is_train=True, with_bbox=True, with_mask=True)),
                phase='train',
                batch_size=batch_size,
                num_workers=num_workers)
    if valid is not None:
        valid = cvtk.ml.mmdetutils.DataLoader(
                    cvtk.ml.mmdetutils.Dataset(datalabel, valid,
                            cvtk.ml.mmdetutils.DataPipeline(is_train=False, with_bbox=True, with_mask=True)),
                    phase='valid',
                    batch_size=batch_size,
                    num_workers=num_workers)
    if test is not None:
        test = cvtk.ml.mmdetutils.DataLoader(
                    cvtk.ml.mmdetutils.Dataset(datalabel, test,
                            cvtk.ml.mmdetutils.DataPipeline(is_train=False, with_bbox=True, with_mask=True)),
                    phase='test',
                    batch_size=batch_size,
                    num_workers=num_workers)
    
    model.train(train, valid, test, epoch=epoch)
    model.save(output_weights)

    train_log_fpath = os.path.splitext(output_weights)[0] + '.train_stats.train.txt'
    valid_log_fpath = os.path.splitext(output_weights)[0] + '.train_stats.valid.txt'
    if os.path.exists(train_log_fpath):
        _plot_log(train_log_fpath)
    if os.path.exists(valid_log_fpath):
        _plot_log(valid_log_fpath)


def _plot_log(log_fpath):    
    log_data = pd.read_csv(log_fpath, sep='\t', header=0, comment='#')
    if 'epoch' in log_data.columns:
        x = 'epoch'
        y = ['loss', 'loss_cls', 'loss_bbox', 'loss_mask', ['acc']]
    else:
        x = 'step'
        y = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/segm_mAP', 'coco/segm_mAP_50']
    cvtk.viz.plot(log_fpath, x=x, y=y, 
                  output=os.path.splitext(log_fpath)[0] + '.png')    


def inference(label, data, model_weights, output, batch_size=4, num_workers=8):
    datalabel = cvtk.ml.data.DataLabel(label)
    
    model = cvtk.ml.mmdetutils.DetRunner(datalabel, os.path.splitext(model_weights)[0] + '.py', model_weights, workspace=output)

    data = cvtk.ml.mmdetutils.DataLoader(
                cvtk.ml.mmdetutils.Dataset(datalabel, data, cvtk.ml.mmdetutils.DataPipeline()),
                phase='inference', batch_size=batch_size, num_workers=num_workers)
    
    pred_outputs = model.inference(data)

    for im in pred_outputs:
        random.seed(1)
        im.draw(layers=['bbox', 'segm'],
                output=os.path.join(output, os.path.basename(im.source)))
    
    imdataset = cvtk.data.ImageDataset(pred_outputs)
    cocodict = imdataset.to_coco()
    cvtk.utils.save_json(cocodict, os.path.join(output, 'instances.coco.json'))


def _train(args):
    train(args.label, args.train, args.valid, args.test, args.output_weights, args.batch_size, args.num_workers, args.epoch)

    
def _inference(args):
    inference(args.label, args.data, args.model_weights, args.output, args.batch_size, args.num_workers)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--label', type=str, required=True)
    parser_train.add_argument('--train', type=str, required=True)
    parser_train.add_argument('--valid', type=str, required=False, default=None)
    parser_train.add_argument('--test', type=str, required=False, default=None)
    parser_train.add_argument('--output_weights', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, default=4)
    parser_train.add_argument('--num_workers', type=int, default=8)
    parser_train.add_argument('--epoch', type=int, default=10)
    parser_train.set_defaults(func=_train)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--label', type=str, required=True)
    parser_inference.add_argument('--data', type=str, required=True)
    parser_inference.add_argument('--model_weights', type=str, required=True)
    parser_inference.add_argument('--output', type=str, required=False)
    parser_inference.add_argument('--batch_size', type=int, default=4)
    parser_inference.add_argument('--num_workers', type=int, default=8)
    parser_inference.set_defaults(func=_inference)

    args = parser.parse_args()
    args.func(args)
    
    
"""
Example Usage:


python __SCRIPTNAME__ train \\
    --label ./data/strawberry/class.txt \\
    --train ./data/strawberry/train/segm.json \\
    --valid ./data/strawberry/valid/segm.json \\
    --test ./data/strawberry/test/segm.json \\
    --output_weights ./output/sb.pth

    
python __SCRIPTNAME__ inference \\
    --label ./data/strawberry/class.txt \\
    --data ./data/strawberry/test/images \\
    --model_weights ./output/sb.pth \\
    --output ./output/pred_results

"""
