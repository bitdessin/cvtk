import argparse
import pprint
import cvtk
import cvtk.ml
import cvtk.data
import cvtk.data.coco
import cvtk.ls


def deploy_model(args):
    cvtk.ml.deploy_model(args.script, backend=args.backend, task=args.task, vanilla=args.vanilla)


def deploy_demoapp(args):
    cvtk.ls.deploy_demoapp(args.project,
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
    cvtk.data.coco.split(input=args.input,
                           output=args.output,
                           ratios=ratios,
                           shuffle=args.shuffle,
                           random_seed=args.random_seed)


def coco_combine(args):
    inputs = args.input.split(',')
    cvtk.data.coco.combine(inputs, output=args.output)


def coco_stats(args):
    pprint.pprint(cvtk.data.coco.stats(args.input))


def coco_crop(args):
    cvtk.data.coco.crop(args.input, output=args.output)


def coco_remove(args):
    images = annotations = categories = None
    if args.images:
        images = args.images.split(',')
        images = [int(i) if i.isdigit() else i for i in images]
    if args.annotations:
        annotations = args.annotations.split(',')
        annotations = [int(i) if i.isdigit() else i for i in annotations]
    if args.categories:
        categories = args.categories.split(',')
        categories = [int(i) if i.isdigit() else i for i in categories]

    cvtk.data.coco.remove(args.input,
                           output=args.output,
                           images=images,
                           categories=categories,
                           annotations=annotations)


def ls_export(args):
    cvtk.ls.export(args.project,
                   output=args.output,
                   format=args.format,
                   url=args.url,
                   token=args.token)


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

    parser_model = subparsers.add_parser('deploy-model')
    parser_model.add_argument('--script', type=str, required=True)
    parser_model.add_argument('--backend', type=str, choices=['torch', 'mmdet'], default='torch')
    parser_model.add_argument('--task', type=str, choices=['cls', 'det', 'segm'], default='cls')
    parser_model.add_argument('--vanilla', action='store_true', default=False)
    parser_model.set_defaults(func=deploy_model)

    parser_demoapp = subparsers.add_parser('deploy-demoapp')
    parser_demoapp.add_argument('--project', type=str, required=True)
    parser_demoapp.add_argument('--source', type=str, required=True)
    parser_demoapp.add_argument('--label', type=str, required=True)
    parser_demoapp.add_argument('--model', type=str, required=True)
    parser_demoapp.add_argument('--weights', type=str, required=True)
    parser_demoapp.add_argument('--vanilla', action='store_true', default=False)
    parser_demoapp.set_defaults(func=deploy_demoapp)

    parser_text_split = subparsers.add_parser('text-split')
    parser_text_split.add_argument('--input', type=str, required=True)
    parser_text_split.add_argument('--output', type=str, required=True)
    parser_text_split.add_argument('--ratios', type=str, default='8:1:1')
    parser_text_split.add_argument('--shuffle', action='store_true')
    parser_text_split.add_argument('--stratify', action='store_true')
    parser_text_split.add_argument('--random_seed', type=int, default=None)
    parser_text_split.set_defaults(func=split)

    parser_coco_split = subparsers.add_parser('coco-split')
    parser_coco_split.add_argument('--input', type=str, required=True)
    parser_coco_split.add_argument('--output', type=str, required=True)
    parser_coco_split.add_argument('--ratios', type=str, default='8:1:1')
    parser_coco_split.add_argument('--shuffle', action='store_true', default=False)
    parser_coco_split.add_argument('--random_seed', type=int, default=None)
    parser_coco_split.set_defaults(func=coco_split)

    parser_coco_combine = subparsers.add_parser('coco-combine')
    parser_coco_combine.add_argument('--input', type=str, required=True)
    parser_coco_combine.add_argument('--output', type=str, required=True)
    parser_coco_combine.set_defaults(func=coco_combine)

    parser_coco_stats = subparsers.add_parser('coco-stats')
    parser_coco_stats.add_argument('--input', type=str, required=True)
    parser_coco_stats.set_defaults(func=coco_stats)

    parser_coco_crop = subparsers.add_parser('coco-crop')
    parser_coco_crop.add_argument('--input', type=str, required=True)
    parser_coco_crop.add_argument('--output', type=str, required=True)
    parser_coco_crop.set_defaults(func=coco_crop)

    parser_coco_remove = subparsers.add_parser('coco-remove')
    parser_coco_remove.add_argument('--input', type=str, required=True)
    parser_coco_remove.add_argument('--output', type=str, required=True)
    parser_coco_remove.add_argument('--images', type=str, required=False)
    parser_coco_remove.add_argument('--categories', type=str, required=False)
    parser_coco_remove.add_argument('--annotations', type=str, required=False)
    parser_coco_remove.set_defaults(func=coco_remove)

    parser_ls_export = subparsers.add_parser('ls-export')
    parser_ls_export.add_argument('--project', type=str, required=True)
    parser_ls_export.add_argument('--output', type=str, required=True)
    parser_ls_export.add_argument('--format', type=str, required=False, default='coco')
    parser_ls_export.add_argument('--url', type=str, required=False, default=None)
    parser_ls_export.add_argument('--token', type=str, required=False, default=None)
    parser_ls_export.set_defaults(func=ls_export)

    parser_ls_backend = subparsers.add_parser('ls-backend')
    parser_ls_backend.add_argument('--project', type=str, required=True)
    parser_ls_backend.add_argument('--source', type=str, required=True)
    parser_ls_backend.add_argument('--label', type=str, required=True)
    parser_ls_backend.add_argument('--model', type=str, default=True)
    parser_ls_backend.add_argument('--weights', type=str, required=True)
    parser_ls_backend.add_argument('--vanilla', action='store_true', default=False)
    parser_ls_backend.set_defaults(func=ls_backend)


    args = parser.parse_args()
    args.func(args)
