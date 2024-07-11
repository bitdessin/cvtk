import os
import sys
import argparse
import cvtk.torch


def create(args):
    if args.type.lower() in ['classification', 'cls']:
        cvtk.torch.create_cls(args.project, args.source)
    else:
        raise ValueError(f'The given project type "{args.type}" is not supported in the current version of cvtk.')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('create')
    parser_train.add_argument('--project', type=str, required=True)
    parser_train.add_argument('--type', type=str, required=True)
    parser_train.add_argument('--source', type=str, default='cvtk')
    parser_train.set_defaults(func=create)

    args = parser.parse_args()
    args.func(args)