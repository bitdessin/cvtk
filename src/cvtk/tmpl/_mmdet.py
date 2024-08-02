import os
from cvtk import ImageDeck
from cvtk.ml.data import DataLabel
from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, MMDETCORE, plot_trainlog


def train(label, train, valid, test, output_weights, batch_size=2, num_workers=8, epoch=10):
    temp_dpath = os.path.splitext(output_weights)[0]

    datalabel = DataLabel(label)
    model = MMDETCORE(datalabel, "__TASKARCH__", None, workspace=temp_dpath)

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
    
    model.train(train, valid, test, epoch=epoch)
    model.save(output_weights)

    if os.path.exists(os.path.splitext(output_weights)[0] + '.train_stats.train.txt'):
        plot_trainlog(os.path.splitext(output_weights)[0] + '.train_stats.train.txt',
                    output=os.path.splitext(output_weights)[0] + '.train_stats.train.png')
    if os.path.exists(os.path.splitext(output_weights)[0] + '.train_stats.valid.txt'):
        plot_trainlog(os.path.splitext(output_weights)[0] + '.train_stats.valid.txt',
                    output=os.path.splitext(output_weights)[0] + '.train_stats.valid.png')    
        

def inference(label, data, model_weights, output, batch_size=4, num_workers=8):
    datalabel = DataLabel(label)
    
    model = MMDETCORE(datalabel, os.path.splitext(model_weights)[0] + '.py', model_weights, workspace=output)

    data = DataLoader(
                Dataset(datalabel, data, DataPipeline()),
                phase='inference', batch_size=batch_size, num_workers=num_workers)
    
    pred_outputs = model.inference(data)

    for im in pred_outputs:
        im.draw(format='bbox+segm',
                output=os.path.join(output, os.path.basename(im.source)))
    
    imdeck = ImageDeck(pred_outputs)
    imdeck.dump(os.path.join(output, 'instances.coco.json'),
                format='coco', datalabel=datalabel.labels)


def _train(args):
    train(args.label, args.train, args.valid, args.test, args.output_weights)

    
def _inference(args):
    inference(args.label, args.data, args.model_weights, args.output)



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
    parser_train.set_defaults(func=_train)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--label', type=str, required=True)
    parser_inference.add_argument('--data', type=str, required=True)
    parser_inference.add_argument('--model_weights', type=str, required=True)
    parser_inference.add_argument('--output', type=str, required=False)
    parser_inference.set_defaults(func=_inference)

    args = parser.parse_args()
    args.func(args)
    
    
"""
Example Usage:


python __SCRIPTNAME__ train \\
    --label ./data/strawberry/class.txt \\
    --train ./data/strawberry/train/__SAMPLEDATA__.json \\
    --valid ./data/strawberry/valid/__SAMPLEDATA__.json \\
    --test ./data/strawberry/test/__SAMPLEDATA__.json \\
    --output_weights ./output/sb.pth

    
python __SCRIPTNAME__ inference \\
    --label ./data/strawberry/class.txt \\
    --data ./data/strawberry/test/images \\
    --model_weights ./output/sb.pth \\
    --output ./output/pred_results

"""
