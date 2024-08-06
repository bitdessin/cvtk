import argparse
import pprint
import cvtk
import cvtk.ml
import cvtk.format
import cvtk.ls



def create(args):
    cvtk.ml.generate_source(args.script, task=args.task, vanilla=args.vanilla)


def app(args):
    cvtk.ml.generate_app(args.project,
                         source=args.source,
                         label=args.label,
                         model=args.model,
                         weights=args.weights,
                         vanilla=args.vanilla)


def split(args):
    ratios = [float(r) for r in args.ratios.split(':')]
    ratios = [r / sum(ratios) for r in ratios]
    subsets = cvtk.ml.split_dataset(data=args.input,
                                    ratios=ratios,
                                    stratify=args.stratify,
                                    shuffle=args.shuffle,
                                    random_seed=args.random_seed)
    for i, subset in enumerate(subsets):
        with open(args.output + '.' + str(i), 'w') as outfh:
            outfh.write('\n'.join(subset) + '\n')


def coco_split(args):
    ratios = [float(r) for r in args.ratios.split(':')]
    ratios = [r / sum(ratios) for r in ratios]
    cvtk.format.coco.split(input=args.input,
                           output=args.output,
                           ratios=ratios,
                           shuffle=args.shuffle,
                           random_seed=args.random_seed)


def coco_combine(args):
    inputs = args.input.split(',')
    cvtk.format.coco.combine(inputs, output=args.output)


def coco_stats(args):
    pprint.pprint(cvtk.format.coco.stats(args.input))


def ls_export(args):
    cvtk.ls.export(args.project,
                   output=args.output,
                   format=args.format,
                   host=args.host,
                   port=args.port,
                   api_key=args.apikey)


def ls_backend(args):
    cvtk.ls.generate_app(args.project,
                         source=args.source,
                         label=args.label,
                         model=args.model,
                         weights=args.weights,
                         vanilla=args.vanilla)



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('create')
    parser_train.add_argument('--script', type=str, required=True)
    parser_train.add_argument('--task', type=str, choices=['cls', 'det', 'segm'], default='cls')
    parser_train.add_argument('--vanilla', action='store_true', default=False)
    parser_train.set_defaults(func=create)

    parser_train = subparsers.add_parser('app')
    parser_train.add_argument('--project', type=str, required=True)
    parser_train.add_argument('--source', type=str, required=True)
    parser_train.add_argument('--label', type=str, required=True)
    parser_train.add_argument('--model', type=str, default=True)
    parser_train.add_argument('--weights', type=str, required=True)
    parser_train.add_argument('--vanilla', action='store_true', default=False)
    parser_train.set_defaults(func=app)

    parser_split_text = subparsers.add_parser('split')
    parser_split_text.add_argument('--input', type=str, required=True)
    parser_split_text.add_argument('--output', type=str, required=True)
    parser_split_text.add_argument('--ratios', type=str, default='8:1:1')
    parser_split_text.add_argument('--shuffle', action='store_true')
    parser_split_text.add_argument('--stratify', action='store_true')
    parser_split_text.add_argument('--random_seed', type=int, default=None)
    parser_split_text.set_defaults(func=split)

    parser_split_text = subparsers.add_parser('coco-split')
    parser_split_text.add_argument('--input', type=str, required=True)
    parser_split_text.add_argument('--output', type=str, required=True)
    parser_split_text.add_argument('--ratios', type=str, default='8:1:1')
    parser_split_text.add_argument('--shuffle', action='store_true', default=False)
    parser_split_text.add_argument('--random_seed', type=int, default=None)
    parser_split_text.set_defaults(func=coco_split)

    parser_split_text = subparsers.add_parser('coco-combine')
    parser_split_text.add_argument('--input', type=str, required=True)
    parser_split_text.add_argument('--output', type=str, required=True)
    parser_split_text.set_defaults(func=coco_combine)

    parser_split_text = subparsers.add_parser('coco-stats')
    parser_split_text.add_argument('--input', type=str, required=True)
    parser_split_text.set_defaults(func=coco_stats)

    parser_split_text = subparsers.add_parser('ls-export')
    parser_split_text.add_argument('--project', type=str, required=True)
    parser_split_text.add_argument('--output', type=str, required=True)
    parser_split_text.add_argument('--format', type=str, required=False, default='coco')
    parser_split_text.add_argument('--host', type=str, required=False, default='localhost')
    parser_split_text.add_argument('--port', type=str, required=False, default=8080)
    parser_split_text.add_argument('--apikey', type=str, required=False, default=None)
    parser_split_text.set_defaults(func=ls_export)

    parser_train = subparsers.add_parser('ls-backend')
    parser_train.add_argument('--project', type=str, required=True)
    parser_train.add_argument('--source', type=str, required=True)
    parser_train.add_argument('--label', type=str, required=True)
    parser_train.add_argument('--model', type=str, default=True)
    parser_train.add_argument('--weights', type=str, required=True)
    parser_train.add_argument('--vanilla', action='store_true', default=False)
    parser_train.set_defaults(func=ls_backend)


    args = parser.parse_args()
    args.func(args)
