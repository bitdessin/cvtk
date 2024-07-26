import argparse
import cvtk.ml.utils



def split_dataset(args):
    ratios = [float(r) for r in args.ratios.split(':')]
    ratios = [r / sum(ratios) for r in ratios]
    if args.type.lower() in ['text', 'txt', 'csv', 'tsv']:
        subsets = cvtk.ml.utils.split_dataset(data=args.input,
                                             ratios=ratios,
                                             balanced=args.balanced,
                                             shuffle=args.shuffle,
                                             random_seed=args.random_seed)

        for i, subset in enumerate(subsets):
            with open(args.output + '.' + str(i), 'w') as outfh:
                outfh.write('\n'.join(subset) + '\n')
    else:
        raise NotImplementedError('The dataset type {} is not supported.'.format(args.type))
    

def create(args):
    cvtk.ml.utils.generate_source(args.script,
                                  task=args.task,
                                  module=args.module)


def app(args):
    cvtk.ml.utils.generate_app(args.project,
                               source=args.source,
                               label=args.label,
                               model=args.model,
                               weights=args.weights,
                               module=args.module)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('create')
    parser_train.add_argument('--script', type=str, required=True)
    parser_train.add_argument('--task', type=str, choices=['cls', 'det', 'segm'], default='cls')
    parser_train.add_argument('--module', type=str, choices=['cvtk', 'vanilla'], default='cvtk')
    parser_train.set_defaults(func=create)

    parser_train = subparsers.add_parser('app')
    parser_train.add_argument('--project', type=str, required=True)
    parser_train.add_argument('--source', type=str, required=True)
    parser_train.add_argument('--label', type=str, required=True)
    parser_train.add_argument('--model', type=str, default=True)
    parser_train.add_argument('--weights', type=str, required=True)
    parser_train.add_argument('--module', type=str, choices=['cvtk', 'vanilla'], default='cvtk')
    parser_train.set_defaults(func=app)

    parser_split_text = subparsers.add_parser('split')
    parser_split_text.add_argument('--input', type=str, required=True)
    parser_split_text.add_argument('--output', type=str, required=True)
    parser_split_text.add_argument('--type', type=str, default='text')
    parser_split_text.add_argument('--ratios', type=str, default='8:1:1')
    parser_split_text.add_argument('--shuffle', action='store_true', default=True)
    parser_split_text.add_argument('--balanced', action='store_true', default=True)
    parser_split_text.add_argument('--random_seed', type=int, default=None)
    parser_split_text.set_defaults(func=split_dataset)

    args = parser.parse_args()
    args.func(args)
