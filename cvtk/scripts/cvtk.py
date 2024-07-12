import os
import sys
import argparse
import cvtk.ml.torch


def create(args):
    cvtk.ml.torch.generate_source(args.project, task=args.type, module=args.module)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('create')
    parser_train.add_argument('--project', type=str, required=True)
    parser_train.add_argument('--type', type=str, default='cls')
    parser_train.add_argument('--module', type=str, default='cvtk')
    parser_train.set_defaults(func=create)

    args = parser.parse_args()
    args.func(args)