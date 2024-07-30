import os
from cvtk.ml.data import DataLabel
from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, CLSCORE, plot_trainlog, plot_cm


def train(label, train, valid, test, output_weights, batch_size=4, num_workers=8, epoch=10):
    temp_dpath = os.path.splitext(output_weights)[0]

    datalabel = DataLabel(label)
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

    model.train(train, valid, test, epoch=epoch)
    model.save(output_weights)

    plot_trainlog(os.path.splitext(output_weights)[0] + '.train_stats.txt',
                  os.path.splitext(output_weights)[0] + '.train_stats.png')
    plot_cm(os.path.splitext(output_weights)[0] + '.test_outputs.txt',
            os.path.splitext(output_weights)[0] + '.test_outputs.cm.png')


def inference(label, data, model_weights, output, batch_size=4, num_workers=8):
    temp_dpath = os.path.splitext(output)[0]

    datalabel = DataLabel(label)
    model = CLSCORE(datalabel, 'resnet18', model_weights, temp_dpath)

    data = DataLoader(
                Dataset(datalabel, data, transform=DataTransform(224, is_train=False)),
                batch_size=batch_size, num_workers=num_workers)
    
    probs = model.inference(data)
    probs.to_csv(output, sep = '\t', header=True, index=True, index_label='image')


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
    parser_train.add_argument('--valid', type=str, required=False)
    parser_train.add_argument('--test', type=str, required=False)
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
    --label ./data/fruits/label.txt \\
    --train ./data/fruits/train.txt \\
    --valid ./data/fruits/valid.txt \\
    --test ./data/fruits/test.txt \\
    --output_weights ./output/fruits.pth

    
python __SCRIPTNAME__ inference \\
    --label ./data/fruits/label.txt \\
    --data ./data/fruits/images \\
    --model_weights ./output/fruits.pth \\
    --output ./output/fruits_results
"""
