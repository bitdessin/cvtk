from __future__ import annotations

import os
import datetime
import shutil
import json
import gc
import pathlib
import ast
import warnings
import filetype
import logging
import gzip
import tempfile
import pickle
import pandas as pd
import numpy as np
import PIL
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import mim
import mmdet
import mmdet.apis
import mmengine.config
import mmengine.runner
import mmdet.evaluation
import cvtk

logger = logging.getLogger(__name__)


class DataPipeline():
    """Generate image preprocessing pipeline

    This class provides the basic image preprocessing pipeline used in MMDetection.

    Args:
        is_train: Whether the pipeline is for training. Default is False.
        with_bbox: Whether the dataset contains bounding boxes.
            Default is True for object detection with bounding boxes only.
        with_mask: Whether the dataset contains masks. Default is False.
    """
    def __init__(self, is_train: bool=False, with_bbox: bool=True, with_mask: bool=False):
        self.__cfg = None

        if is_train:
            self.__cfg = [
                dict(type='LoadImageFromFile',
                     backend_args=None),
                dict(type='LoadAnnotations',
                     with_bbox=with_bbox,
                     with_mask=with_mask),
                dict(type='Resize',
                     scale=(1333, 800),
                     keep_ratio=True),
                dict(type='RandomFlip',
                     prob=0.5),
                dict(type='PackDetInputs')
            ]
        else:
            self.__cfg = [
                dict(type='LoadImageFromFile',
                     backend_args=None),
                dict(type='LoadAnnotations',
                     with_bbox=with_bbox,
                     with_mask=with_mask),
                dict(type='Resize',
                     scale=(1333, 800),
                     keep_ratio=True),
                dict(
                     type='PackDetInputs',
                     meta_keys=('img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor'))
            ]

    @property
    def cfg(self):
        return self.__cfg


class Dataset():
    """Generate dataset configuration

    This function generates the dataset configuration for MMDetection.

    Args:
        datalabel: A DataLabel class object.
        dataset: A path to a COCO format file with extension '.json',
            a path to a directory containing images,
            a path to an image file, or a list of paths to image files.
            Note that, for training, validation, and test, the COCO format file is required.
        pipeline: A DataPipeline class object.
        repeat_dataset: Whether to repeat the dataset. Default is False.
            Use the repeated dataset for training will be faster in some architecture.
        image_root: Base directory for resolving COCO image `file_name` paths.
            If None, image paths are resolved relative to the COCO annotation file directory.
    """
    def __init__(
        self,
        datalabel: cvtk.ml.data.DataLabel,
        dataset: str|list[str]|dict|None,
        pipeline: DataPipeline|None=None,
        repeat_dataset: bool=False,
        image_root: str|None=None,
    ):
        self.__cfg = None
        if pipeline is None:
            pipeline = DataPipeline()

        if dataset is None:
            self.__cfg = None
        elif isinstance(dataset, str) and dataset.endswith('.json'):
            self.__check_coco_format(datalabel, dataset)
            ann_file = os.path.abspath(dataset)
            if image_root is None:
                data_root = os.path.dirname(ann_file)
            else:
                data_root = os.path.abspath(image_root)
            self.__cfg = dict(
                metainfo=dict(classes=datalabel.labels),
                type='CocoDataset',
                data_root=data_root,
                data_prefix=dict(img=''),
                ann_file=ann_file,
                pipeline=pipeline.cfg,
                filter_cfg=dict(filter_empty_gt=True, min_size=0),
            )
            if repeat_dataset:
                self.__cfg = dict(
                    type='RepeatDataset',
                    times=1,
                    dataset=self.__cfg,
                )
        elif isinstance(dataset, (list, tuple)):
            self.__cfg = dict(
                metainfo=dict(classes=datalabel.labels),
                type='CocoDataset',
                pipeline=pipeline.cfg,
                data_root=dataset,
            )
        elif isinstance(dataset, str):
            self.__cfg = dict(
                metainfo=dict(classes=datalabel.labels),
                type='CocoDataset',
                pipeline=pipeline.cfg,
                data_root=os.path.abspath(dataset),
            )
        elif isinstance(dataset, dict):
            self.__cfg = dataset
        else:
            raise TypeError(f'Invalid type: {type(dataset)}')
    
    @property
    def cfg(self):
        return self.__cfg
    

    def __check_coco_format(self, datalabel, ann_fpath):
        with open(ann_fpath) as fh:
            cocodict = json.load(fh)
        
        # check coco format
        if 'images' not in cocodict:
            raise ValueError(f'Invalid COCO format: {ann_fpath}. No "images" field.')
        if 'annotations' not in cocodict:
            raise ValueError(f'Invalid COCO format: {ann_fpath}. No "annotations" field.')
        if 'categories' not in cocodict:
            raise ValueError(f'Invalid COCO format: {ann_fpath}. No "categories" field.')
        if 'info' not in cocodict:
            warnings.warn(f'COCO format file without "info" field may cause processing errors in some versions of MMDetection. It is recommended to add an "info" field into the file: {ann_fpath}.')

        # check file contains data
        if (len(cocodict['images']) == 0):
            raise ValueError(f'No images in "images" field.')
        if (len(cocodict['annotations']) == 0):
            raise ValueError(f'No annotations in "annotations" field.')
        if (len(cocodict['categories']) == 0):
            raise ValueError(f'No categories in "categories" field.')

        # check label names
        coco_classes = [cat['name'] for cat in sorted(cocodict['categories'], key=lambda x: x['id'])]
        input_classes = datalabel.labels
        if len(coco_classes) != len(input_classes):
            raise ValueError(f'The number of classes in annotations ({len(coco_classes)}) is different from that in datalabel ({len(input_classes)}).')
        for i in range(len(coco_classes)):
            if coco_classes[i] != input_classes[i]:
                raise ValueError(f'Class names are different between annotations and datalabel at index {i}: "{coco_classes[i]}" (in annotations) vs "{input_classes[i]}" (in datalabel).')
        


class DataLoader():
    """Generate dataloader configuration

    This function generates the dataloader configuration for MMDetection.

    Args:
        dataset: A Dataset class object.
        phase: The purpose of DataLoader usage. It shold be specified as one
            'train', 'valie', 'test', and 'inference'.
        batch_size (int): Batch size.
        num_workers (int): Number of threads for data preprocessing and loading.
    """
    def __init__(self, dataset: Dataset|None=None, phase: str='inference', batch_size: int=4, num_workers: int=4):
        self.__cfg = None

        if dataset is None:
            dataset = Dataset(cvtk.ml.data.DataLabel([]), None)

        dataset_cfg = dataset.cfg
        base_dataset_cfg = self.__unwrap_dataset_cfg(dataset_cfg)

        metrics = ['bbox']
        if base_dataset_cfg is not None:
            if 'pipeline' in base_dataset_cfg:
                for pp in base_dataset_cfg['pipeline']:
                    if pp['type'] == 'LoadAnnotations':
                        if 'with_mask' in pp and pp['with_mask']:
                            metrics.append('segm')
        
        if phase == 'train':
            if dataset_cfg is None:
                raise ValueError('The dataset configuration is required for training, but got None.')
            else:
                self.__cfg = dict(
                    dataset_type='CocoDataset',
                    train_dataloader=dict(
                        batch_size=batch_size,
                        num_workers=num_workers,
                        dataset=dataset_cfg,
                    ),
                    train_cfg = dict(
                        type='EpochBasedTrainLoop',
                        max_epochs=12,
                        val_interval=1,
                    ),
                )
        elif phase == 'valid':
            if dataset_cfg is None:
                self.__cfg = dict(
                        val_dataloader=None,
                        val_cfg=None,
                        val_evaluator=None)
            else:
                if base_dataset_cfg is None or ('ann_file' not in base_dataset_cfg):
                    raise ValueError('Validation requires COCO annotation dataset with ann_file.')
                self.__cfg = dict(
                        val_dataloader=dict(
                            _delete_=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            dataset=dataset_cfg,
                            drop_last=False,
                            sampler=dict(type='DefaultSampler', shuffle=False)
                        ),
                        val_cfg = dict(
                            _delete_=True,
                            type='ValLoop'),
                        val_evaluator = dict(
                            _delete_=True,
                            type='CocoMetric',
                            ann_file=base_dataset_cfg['ann_file'],
                            metric=metrics,
                            backend_args=None
                        )
                    )
        elif phase == 'test':
            if dataset_cfg is None:
                self.__cfg = dict(
                        test_dataloader=None,
                        test_cfg=None,
                        test_evaluator=None)
            else:
                if base_dataset_cfg is None or ('ann_file' not in base_dataset_cfg):
                    raise ValueError('Testing requires COCO annotation dataset with ann_file.')
                self.__cfg = dict(
                        test_dataloader=dict(
                            _delete_=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            dataset=dataset_cfg,
                            drop_last=False,
                            sampler=dict(type='DefaultSampler', shuffle=False)
                        ),
                        test_cfg = dict(
                            _delete_=True,
                            type='TestLoop'),
                        test_evaluator = dict(
                            _delete_=True,
                            type='CocoMetric',
                            ann_file=base_dataset_cfg['ann_file'],
                            metric=metrics,
                            backend_args=None
                        )
                    )
        else: # other cases, e.g., inference
            self.__cfg = dict(test_dataloader=dict(
                        _delete_=True,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=False),
                        dataset=dataset_cfg))


    def __unwrap_dataset_cfg(self, dataset_cfg):
        while (
            isinstance(dataset_cfg, dict)
            and ('dataset' in dataset_cfg)
            and isinstance(dataset_cfg['dataset'], dict)
        ):
            dataset_cfg = dataset_cfg['dataset']
        return dataset_cfg



    @property
    def cfg(self):
        return self.__cfg        


class DetRunner():
    """A class for object detection and instance segmentation

    This class provides user-friendly APIs for object detection and instance segmentation
    using MMDetection.
    There are four main methods are implemented in this class:
    :func:`train <cvtk.ml.mmdetutils.DetRunner.train>`,
    :func:`test <cvtk.ml.mmdetutils.DetRunner.test>`,
    :func:`save <cvtk.ml.mmdetutils.DetRunner.save>`,
    :func:`inference <cvtk.ml.mmdetutils.DetRunner.inference>`.
    The :func:`train <cvtk.ml.mmdetutils.DetRunner.train>` method is used for training the model
    and perform validation and test if validation and test data are provided.
    The :func:`test <cvtk.ml.mmdetutils.DetRunner.test>` method is used for testing the model with test data.
    In general, the performance test is performed automatically after the training,
    but user can also run the test independently from the training process with
    the :func:`test <cvtk.ml.mmdetutils.DetRunner.test>` method.
    The :func:`save <cvtk.ml.mmdetutils.DetRunner.save>` method is used for saving the model weights,
    configuration (design of model architecture), training log (e.g., mAP and loss per epoch), and test results.
    The :func:`inference <cvtk.ml.mmdetutils.DetRunner.inference>` method is used for running inference
    with the trained model.
    The detailed usage of each method is described in the method documentation.


    Run `mim search mmdet --model "faster r-cnn"` to set the pre-defined configuration for `cfg`.

    Args:
        datalabel: A :class:`DataLabel <cvtk.ml.data.DataLabel>` class object,
            a path to a file containing class labels,
            or a list of class labels.
        cfg: A path to a file containing model configuration (usually with extension '.py'),
            a dictionary of a model configuration,
            or a keyword of configuration pre-defined in MMDetection.
            The pre-defined configuration can be found from MMDetection GitHub repository
            or list up with the `mim` command (e.g., `mim search mmdet --model "faster r-cnn"`).
        weights: A path to a file containing model weights (usually with extension '.pth').
            If `None`, the function will download the pre-trained model weights
            from the MMDetection repository,
            or use the random weights if the download is not available.
        workspace: A path to a directory for storing the intermediate files.
            If not specified, this function creates a temporary directory in the OS temporary directory
            and removes it after the process is completed.
        seed: A seed for model training.

    Examples:
        >>> from cvtk.ml.data import DataLabel
        >>> from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, DetRunner
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'stem'])
        >>> cfg = 'faster_rcnn_r50_fpn_1x_coco'
        >>> weights = None # download from MMDetection repository
        >>> workspace = '/path/to/workspace'
        >>>
        >>> model = DetRunner(datalabel, cfg, weights, workspace)
        >>> 
        >>> train = DataLoader(Dataset(datalabel, '/path/to/train/coco.json'), 'train')
        >>> model.train(train, epoch=10)
        >>> model.save('/path/to/model.pth')
    """
    def __init__(self,
                 datalabel: cvtk.ml.data.DataLabel|str|list[str]|tuple[str],
                 cfg: str|dict,
                 weights: str|None=None,
                 workspace=None,
                 seed=None):
        self.task_type = 'det'
        if not(datalabel is None and cfg is None):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.datalabel = self.__init_datalabel(datalabel)
            self.cfg = self.__init_cfg(cfg, weights, seed)
            self.model = None
            self.__tempdir_obj, self.workspace = self.__init_tempdir(workspace)
            self.mmdet_log_dpath = None
            self.test_stats = None
    

    def __del__(self):
        try:
            if hasattr(self, '__tempdir_obj') and (self.model is not None):
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
            if hasattr(self, '__tempdir_obj') and (self.__tempdir_obj is not None):
                self.__tempdir_obj.cleanup()
        except:
            logger.info(f'The temporary directory (`{self.workspace}`) created by cvtk '
                        f'cannot be removed automatically. Please remove it manually.')


    def __init_datalabel(self, datalabel):
        if isinstance(datalabel, cvtk.ml.data.DataLabel):
            pass
        elif isinstance(datalabel, str) or isinstance(datalabel, list) or isinstance(datalabel, tuple):
            datalabel = cvtk.ml.data.DataLabel(datalabel)
        else:
            raise TypeError('Invalid type: {}'.format(type(datalabel)))
        return datalabel


    def __init_cfg(self, cfg, weights, seed):
        if cfg is None or cfg == '':
            raise TypeError(f'cvtk requires a configuration file to build models. '
                            f'Set up a path to a configuration file, a dictionary of configuration or '
                            f'a string of pre-defined configuration. '
                            f'The pre-defined configuration can be found from MMDetection GitHub repository or '
                            f'list up with `mim search mmdet --valid-config` command.')
        
        chk = None
        if isinstance(cfg, str):
            if not os.path.exists(cfg):
                cache_dpath = os.path.join(os.path.expanduser('~'), '.cache', 'mim')
                chk = mim.commands.download(package='mmdet', configs=[cfg])[0]
                cfg = os.path.join(cache_dpath, cfg + '.py')
                chk = os.path.join(cache_dpath, chk)
            cfg = mmengine.config.Config.fromfile(cfg)
        elif isinstance(cfg, dict):
            cfg = mmengine.config.Config(cfg)
        else:
            raise TypeError(f'The configuration is expected to be a path to a configuration file, '
                            f'a dictionary of configuration, or a string of pre-defined configuration, '
                            f'but got {cfg=} ({type(cfg)}).')
    
        if weights is None:
            if chk is not None:
                cfg.load_from = chk
        else:
            if os.path.exists(weights):
                cfg.load_from = weights
            else:
                raise FileNotFoundError(f'The file of model weights ({weights}) does not exist. '
                                        f'Please check the file path or the internet connection and try again.')

        cfg.launcher = 'none'
        cfg.resume = False
        cfg = self.__set_datalabel(cfg, self.datalabel.labels)
        cfg.seed = seed if seed is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        return  cfg


    def __set_datalabel(self, cfg, class_labels):
        def __set_cl(cfg, class_labels):
            for cfg_key in cfg:
                if isinstance(cfg[cfg_key], dict):
                    __set_cl(cfg[cfg_key], class_labels)
                elif isinstance(cfg[cfg_key], (list, tuple)):
                    if len(cfg[cfg_key]) > 0 and isinstance(cfg[cfg_key][0], dict):
                        for elem in cfg[cfg_key]: 
                            __set_cl(elem, class_labels)
                else:
                    if cfg_key == 'classes':
                        cfg[cfg_key] = class_labels
                    elif cfg_key == 'num_classes' or cfg_key == 'num_things_classes':
                        cfg[cfg_key] = len(class_labels)
            return cfg
        
        cfg.data_root = ''
        cfg.merge_from_dict(dict(metainfo = dict(classes=class_labels)))
        cfg.model = __set_cl(cfg.model, class_labels)
        # for RetinaNet: ResNet: init_cfg and pretrained cannot be specified at the same time
        if 'pretrained' in cfg.model:
            del cfg.model['pretrained']
        return cfg
    

    def __init_tempdir(self, workspace):
        tempd = None
        if workspace is None:
            tempd = tempfile.TemporaryDirectory()
            self.cfg.work_dir = tempd.name
        else:
            if not os.path.exists(workspace):
                os.makedirs(workspace)
            self.cfg.work_dir = workspace
        return tempd, self.cfg.work_dir


    def train(
        self,
        train: DataLoader,
        valid: DataLoader|None=None,
        test: DataLoader|None=None,
        epoch: int=20,
        optimizer: dict|str|None=None,
        scheduler: dict|str|None=None
    ):
        """Perform model training

        The model can be trained with just the training data,
        but it is highly recommended to also provide validation and test data
        to thoroughly evaluate the model's performance.
        If validation data is provided,
        the model's performance will be evaluated after each epoch,
        and the metrics will be saved in the workspace.
        This allows the user to monitor the model's progress and performance
        throughout the training process.
        Additionally, if test data is provided,
        the model will undergo a final evaluation at the end of training,
        and the test results will also be saved in the workspace.
        The test can also be performed independently from the training process,
        seed the :func:`test <cvtk.ml.mmdetutils.DetRunner.test>` method for more details.

        Args:
            train: A DataLoader class object.
            valid: A DataLoader class object or None.
            test: A DataLoader class object or None.
            epoch: The number of epochs.
            optimizer: A dictionary of string indicating optimizer for training.
            scheduler: A dictionary of string indicating scheduler for training.

    Examples:
        >>> from cvtk.ml.data import DataLabel
        >>> from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, DetRunner
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'stem'])
        >>> cfg = 'faster_rcnn_r50_fpn_1x_coco'
        >>> weights = None # download from MMDetection repository
        >>> workspace = '/path/to/workspace'
        >>>
        >>> model = DetRunner(datalabel, cfg, weights, workspace)
        >>> 
        >>> train = DataLoader(Dataset(datalabel, '/path/to/train/coco.json'), 'train')
        >>> model.train(train, epoch=10)
        >>> model.save('/path/to/model.pth')
        >>>
        >>>
        >>> train = DataLoader(Dataset(datalabel, '/path/to/train/coco.json'), 'train')
        >>> valid = DataLoader(Dataset(datalabel, '/path/to/valid/coco.json'), 'valid')
        >>> test = DataLoader(Dataset(datalabel, '/path/to/test/coco.json'), 'test')
        >>> model.train(train, valid, test, epoch=10)
        >>> model.save('/path/to/model.pth')
        """
        self.cfg.device = self.device
        
        # training params
        self.__set_optimizer(optimizer)
        self.__set_scheduler(scheduler)
        
        # datasets
        self.cfg.merge_from_dict(train.cfg)
        
        self.cfg.train_cfg.max_epochs = epoch
        if valid is None:
            valid = DataLoader(None, 'valid')
        self.cfg.merge_from_dict(valid.cfg)
        _test_none = DataLoader(None, 'test')
        self.cfg.merge_from_dict(_test_none.cfg) # test after training
        self.cfg.default_hooks.checkpoint.interval = 1000

        # training
        runner = mmengine.runner.Runner.from_cfg(self.cfg)
        self.mmdet_log_dpath = os.path.join(self.workspace, runner.timestamp)
        runner.train()
        del runner
        torch.cuda.empty_cache()
        gc.collect()
        self.save(os.path.join(self.workspace, 'last_checkpoint.pth'))

        # test
        if test is not None:
            #self.cfg.merge_from_dict(test.cfg)
            self.cfg.load_from = os.path.join(self.workspace, 'last_checkpoint.pth')
            self.test_stats = self.test(test)


    def __set_optimizer(self, optimizer):
        if optimizer is not None:
            if isinstance(optimizer, dict):
                opt_dict = optimizer
            elif isinstance(optimizer, str) and optimizer.replace(' ', '') != '':
                if optimizer[0] != '{' and optimizer[0:4] != 'dict':
                    optimizer = 'dict(' + optimizer + ')'
                opt_dict = self.__safe_eval(optimizer)
                if not isinstance(opt_dict, dict):
                    raise TypeError(f'Optimizer string must resolve to dict, got {type(opt_dict)}.')
            else:
                raise TypeError(f'Invalid optimizer type: {type(optimizer)}')
            self.cfg.optim_wrapper = dict(optimizer=opt_dict, type='OptimWrapper')
    

    def __set_scheduler(self, scheduler):
        if scheduler is not None:
            if isinstance(scheduler, list) or isinstance(scheduler, tuple):
                scheduler_dict = scheduler
            elif isinstance(scheduler, str) and scheduler.replace(' ', '') != '':
                if scheduler[0] == '[':
                    pass
                else:
                    if scheduler[0] == '{' or scheduler[0:4] == 'dict':
                        scheduler = '[' + scheduler + ']'
                    else:
                        scheduler = '[dict(' + scheduler + ')]'
                scheduler_dict = self.__safe_eval(scheduler)
                if not isinstance(scheduler_dict, (list, tuple)):
                    raise TypeError(f'Scheduler string must resolve to list/tuple, got {type(scheduler_dict)}.')
            else:
                raise TypeError(f'Invalid scheduler type: {type(scheduler)}')
            self.cfg.param_scheduler = scheduler_dict


    def __safe_eval(self, expr):
        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                val = _eval_node(node.operand)
                if not isinstance(val, (int, float)):
                    raise ValueError('Unary operators are only supported for numeric values.')
                return +val if isinstance(node.op, ast.UAdd) else -val

            if isinstance(node, ast.List):
                return [_eval_node(e) for e in node.elts]

            if isinstance(node, ast.Tuple):
                return tuple(_eval_node(e) for e in node.elts)

            if isinstance(node, ast.Dict):
                out = {}
                for k, v in zip(node.keys, node.values):
                    key = _eval_node(k)
                    out[key] = _eval_node(v)
                return out

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id != 'dict':
                    raise ValueError('Only dict(...) calls are supported in config strings.')

                out = {}
                if len(node.args) > 1:
                    raise ValueError('dict(...) supports at most one positional argument here.')
                if len(node.args) == 1:
                    base = _eval_node(node.args[0])
                    if not isinstance(base, dict):
                        raise ValueError('Positional argument to dict(...) must resolve to dict.')
                    out.update(base)
                for kw in node.keywords:
                    if kw.arg is None:
                        raise ValueError('dict(**kwargs) is not supported in config strings.')
                    out[kw.arg] = _eval_node(kw.value)
                return out

            raise ValueError(f'Unsupported syntax in config string: {ast.dump(node)}')

        tree = ast.parse(expr, mode='eval')
        return _eval_node(tree.body)


    def __unwrap_dataset_cfg(self, dataset_cfg):
        while (
            isinstance(dataset_cfg, dict)
            and ('dataset' in dataset_cfg)
            and isinstance(dataset_cfg['dataset'], dict)
        ):
            dataset_cfg = dataset_cfg['dataset']
        return dataset_cfg


    def test(
        self,
        test: DataLoader
    ) -> dict:
        """Validate the model with test data
        
        This method is used to validate the model with test data.
        The test data should be COCO format file containing the annotations
        and converted to a dictionary withs :func:`DataLoader <cvtk.ml.mmdetutils.DataLoader>`.
        The predicted annotations of test data will be stored in the workspace
        with the names of :file:`test_outputs.pkl` in MMDetection format and
        :file:`test_outputs.coco.json` in COCO format.
        The performance metrics (e.g., mAP) will be returned as a dictionary.

        Args:
            test: A DataLoader class object configured for test phase.

        Examples:
        >>> from cvtk.ml.data import DataLabel
        >>> from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, DetRunner
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'stem'])
        >>> cfg = 'faster_rcnn_r50_fpn_1x_coco'
        >>> weights = '/path/to/model.pth'
        >>>
        >>> model = DetRunner(datalabel, cfg, weights, workspace)
        >>> 
        >>> test = DataLoader(Dataset(datalabel, '/path/to/test/coco.json'), 'test')
        >>> metrics = model.test(test)
        >>> print(metrics)
        """
        if not isinstance(test, DataLoader):
            raise TypeError(f'`test` must be a DataLoader object, but got {type(test)}.')

        self.cfg.merge_from_dict(test.cfg)
        runner = mmengine.runner.Runner.from_cfg(self.cfg)

        test_outputs = os.path.join(self.workspace, 'test_outputs.pkl')
        runner.test_evaluator.metrics.append(mmdet.evaluation.DumpDetResults(out_file_path=test_outputs))
        runner.test()

        with open(test_outputs, 'rb') as infh:
            pred_outputs = pickle.load(infh)

        input_image_name_map = self.__get_input_image_name_map()

        cocodict = {'images': [], 'annotations': [], 'categories': []}
        for cate in self.datalabel.labels:
            cocodict['categories'].append({
                'id': self.datalabel[cate],
                'name': cate
            })

        annid = 0
        for po in pred_outputs:
            img_id = po['img_id']
            img_path = input_image_name_map.get(img_id, po.get('img_path'))
            if img_path in (None, ''):
                img_path = f'image_{img_id}'
            cocodict['images'].append({
                'id': img_id,
                'file_name': img_path,
                'width': po['ori_shape'][1] if 'ori_shape' in po else None,
                'height': po['ori_shape'][0] if 'ori_shape' in po else None
            })

            if po.get('pred_instances') is not None:
                imsize = (po['ori_shape'][1], po['ori_shape'][0]) if 'ori_shape' in po else None
                imann = self.__format_mmdet_output(img_path, po.get('pred_instances'), cutoff=0, imsize=imsize)
                for ann in imann.annotations:
                    annid += 1
                    bbox_xywh = ann.bbox.to_xywh() if ann.bbox else None
                    cocodict['annotations'].append({
                        'id': annid,
                        'image_id': img_id,
                        'category_id': self.datalabel[ann.label],
                        'score': ann.score,
                        'bbox': bbox_xywh,
                        'area': ann.area,
                        'iscrowd': 0
                    })
                    if ann.segm is not None:
                        cocodict['annotations'][-1]['segmentation'] = ann.segm.to_rle()

        cvtk.utils.save_json(cocodict,
                             os.path.splitext(test_outputs)[0] + '.coco.json', indent=4, ensure_ascii=False)

        iou_type = 'bbox'
        test_dataset_cfg = self.__unwrap_dataset_cfg(self.cfg.test_dataloader.dataset)
        for pp in test_dataset_cfg.get('pipeline', []):
            if pp['type'] == 'LoadAnnotations':
                if 'with_mask' in pp and pp['with_mask']:
                    iou_type = 'segm'

        self.test_stats = cvtk.data.coco.calc_stats(self.cfg.test_evaluator.ann_file,
                                     os.path.splitext(test_outputs)[0] + '.coco.json',
                                     image_by='file_name',
                                     category_by='name',
                                     iouType=iou_type)
        

        del runner
        torch.cuda.empty_cache()
        gc.collect()
        self.save(os.path.join(self.workspace, 'last_checkpoint.pth'))

        return self.test_stats
    

    

    def save(
        self,
        output: str
    ):
        """Save the model

        Save the model. If training metrics and test results,
        usually generated from training process,
        are exists, they will be save in the same name of weights but
        with the different suffixes.

        Args:
            output: A path to store the model weights and configuration.
        
        """
        if not output.endswith('.pth'):
            output += '.pth'
        if not os.path.exists(os.path.dirname(output)):
            if os.path.dirname(output) != '':
                os.makedirs(os.path.dirname(output))

        checkpoint_ref = os.path.join(self.workspace, 'last_checkpoint')
        if os.path.exists(checkpoint_ref):
            with open(checkpoint_ref) as chkf:
                checkpoint_path = chkf.readline().strip()
            shutil.copy2(checkpoint_path, output)
        elif getattr(self.cfg, 'load_from', None) is not None and os.path.exists(self.cfg.load_from):
            shutil.copy2(self.cfg.load_from, output)
        else:
            raise FileNotFoundError(f'No checkpoint found to save. Expected {checkpoint_ref} or valid cfg.load_from.')
        
        cfg_fpath = os.path.splitext(output)[0] + '.py'
        self.cfg.dump(cfg_fpath)
        self.datalabel.save(os.path.splitext(output)[0] + '.dl.txt')

        self.__write_trainlog(os.path.splitext(output)[0] + '.train_stats')

        if self.test_stats is not None:
            with open(os.path.join(os.path.splitext(output)[0] + '.test_stats.json'), 'w') as outfh:
                json.dump(self.test_stats, outfh, indent=4, ensure_ascii=False)


    def __write_trainlog(self, output_prefix=None):
        train_log = []
        valid_log = []

        if self.mmdet_log_dpath is None:
            warnings.warn('Training log directory is not set. Skip writing training logs.')
            return

        log_fpath = os.path.join(self.mmdet_log_dpath, 'vis_data', 'scalars.json')
        if os.path.exists(log_fpath):
            with open(log_fpath) as fh:
                for log_data in fh:
                    if 'coco/bbox_mAP' in log_data:
                        valid_log.append(log_data)
                    else:
                        train_log.append(log_data)
            
        if len(train_log) > 0:
            (pd.DataFrame(json.loads('[' + ','.join(train_log) + ']'))
                .groupby('epoch')
                .mean()
                .drop(columns=['iter', 'step'])
                .to_csv(output_prefix + '.train.txt', header=True, index=True, sep='\t'))
        if len(valid_log) > 0:
            (pd.DataFrame(json.loads('[' + ','.join(valid_log) + ']'))
                .to_csv(output_prefix + '.valid.txt', header=True, index=False, sep='\t'))


    def inference(
        self,
        data: DataLoader|str|list[str],
        cutoff: float=0.5
    ) -> cvtk.data.ImageDataset:
        """Perform model inference on images.
        
        Run inference on provided images and return results as an ImageDataset.
        Each image in the dataset contains InstanceAnnotation objects with predictions
        (bounding boxes, segmentations, scores).
        
        Args:
            data: A DataLoader object, a path to an image file, or a list of image paths.
            cutoff (float): Score threshold for filtering predictions. Default 0.5.
            
        Returns:
            ImageDataset: Collection of ImageRecord objects with predictions.
            
        Examples:
            >>> test_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']
            >>> dataset = model.inference(test_images)
            >>> for record in dataset:
            >>>     bbox_img_fpath = os.path.splitext(str(record.source))[0] + '.bbox.png'
            >>>     record.draw(layers=['bbox', 'segm'], output=bbox_img_fpath)
            >>> dataset.to_coco('predictions.json')  # Export as COCO format
        """
        original_data = data
        input_images = []
        input_image_sources = []
        
        if not isinstance(data, DataLoader):
            if not isinstance(data, str) and not isinstance(data, list):
                raise TypeError(f'`data` must be DataLoader, str, or list[str], but got {type(data)}.')
            if isinstance(data, list) and any(not isinstance(p, str) for p in data):
                raise TypeError('`data` list input must contain only string image paths.')
            # dataloader is not given, set minimum resource for inference
            data = DataLoader(
                    Dataset(self.datalabel, data, pipeline=DataPipeline()),
                    phase='inference', batch_size=1, num_workers=1)
        
        # test dataloader defined by mmdet
        self.cfg.merge_from_dict(data.cfg)
        ds_cfg = self.cfg.test_dataloader.dataset
        if ('type' in ds_cfg) and (ds_cfg['type'] == 'RepeatDataset') and ('dataset' in ds_cfg):
            ds_cfg = ds_cfg['dataset']

        data_dpath = ds_cfg['data_root'] if ('data_root' in ds_cfg) else ''
        image_root = None
        if ('ann_file' in ds_cfg) and (ds_cfg['ann_file'] not in (None, '')):
            data_dpath = ds_cfg['ann_file']
            image_root = ds_cfg['data_root'] if ('data_root' in ds_cfg) else None
            with open(ds_cfg['ann_file']) as infh:
                cocodict = json.load(infh)
            input_image_sources = [im['file_name'] for im in cocodict.get('images', [])]

        input_images = self.__load_images(data_dpath, image_root=image_root)
        if len(input_image_sources) != len(input_images):
            input_image_sources = input_images
        elif isinstance(original_data, (str, list)) and ('ann_file' not in ds_cfg or ds_cfg.get('ann_file') in (None, '')):
            input_image_sources = input_images

        if self.model is None:
            self.model = mmdet.apis.init_detector(self.cfg, self.cfg.load_from, device=self.device)
        pred_outputs = mmdet.apis.inference_detector(self.model, input_images)
        
        # Format predictions as ImageRecord objects
        records = []
        for target_image, output in zip(input_image_sources, pred_outputs):
            # Extract image size from MMDetection output
            imsize = None
            if hasattr(output, 'ori_shape') and output.ori_shape is not None:
                imsize = (output.ori_shape[1], output.ori_shape[0])
            record = self.__format_mmdet_output(target_image, output.pred_instances, cutoff, imsize=imsize)
            records.append(record)
        
        return cvtk.data.ImageDataset(records=records)
    
    
    def __load_images(self, dataset, image_root: str|None=None):
        x = []
        if isinstance(dataset, str):
            if os.path.isfile(dataset):
                if filetype.is_image(dataset):
                    x = [dataset]
                else:
                    if dataset.endswith('.gz') or dataset.endswith('.gzip'):
                        trainfh = gzip.open(dataset, 'rt')
                    else:
                        trainfh = open(dataset, 'r')
                    if dataset.endswith('.json') or dataset.endswith('.json.gz') or dataset.endswith('.json.gzip'):
                        cocofh = json.load(trainfh)
                        if image_root is None:
                            image_base = os.path.dirname(os.path.abspath(dataset))
                        else:
                            image_base = os.path.abspath(image_root)
                        for im in cocofh['images']:
                            im_path = im['file_name']
                            if im_path in (None, ''):
                                raise ValueError('Invalid COCO image entry: missing "file_name".')
                            if not os.path.isabs(im_path):
                                im_path = os.path.join(image_base, im_path)
                            x.append(im_path)
                    else:
                        x = []
                        for line in trainfh:
                            words = line.rstrip().split('\t')
                            if len(words) == 0 or words[0] == '':
                                continue
                            x.append(words[0])
                    trainfh.close()
            elif os.path.isdir(dataset):
                for root, dirs, files in os.walk(dataset):
                    for f in files:
                        if filetype.is_image(os.path.join(root, f)):
                            x.append(os.path.join(root, f))
        elif isinstance(dataset, list) or isinstance(dataset, tuple):
            x = dataset
        
        if len(x) == 0:
            raise ValueError(f'No image files found in the given dataset ({dataset}).')

        return x
    
    
    def __format_mmdet_output(self, im_fpath, pred_instances, cutoff, imsize=None):
        """Format MMDetection predictions to ImageRecord with InstanceAnnotation objects.
        
        Converts MMDetection model predictions into cvtk data structures for
        consistent handling across the library.
        
        Args:
            im_fpath (str): Image file path or identifier.
            pred_instances: MMDetection pred_instances containing bboxes, labels, scores, masks.
            cutoff (float): Score threshold for filtering predictions.
            imsize (tuple[int, int]|None): Image size as (width, height). Default None.
            
        Returns:
            ImageRecord: Record with list of InstanceAnnotation objects (predictions).
        """
        # Extract bounding boxes, labels, and scores
        if 'bboxes' in pred_instances:
            if isinstance(pred_instances, dict):
                pred_bboxes = pred_instances['bboxes'].detach().cpu().numpy().tolist()
                pred_labels = pred_instances['labels'].detach().cpu().numpy().tolist()
                pred_scores = pred_instances['scores'].detach().cpu().numpy().tolist()
            else:
                pred_bboxes = pred_instances.bboxes.detach().cpu().numpy().tolist()
                pred_labels = pred_instances.labels.detach().cpu().numpy().tolist()
                pred_scores = pred_instances.scores.detach().cpu().numpy().tolist()
        else:
            pred_bboxes = []
            pred_labels = []
            pred_scores = []
        
        # Extract masks (format is consistent within a single output: all RLE or all arrays)
        pred_masks = [None] * len(pred_bboxes)
        if 'masks' in pred_instances:
            if isinstance(pred_instances, dict):
                masks_raw = pred_instances['masks']
            else:
                masks_raw = pred_instances.masks.detach().cpu().numpy()
            
            # Determine mask format from first mask and process all consistently
            if len(masks_raw) > 0:
                is_rle_format = isinstance(masks_raw[0], dict)
                for i, mask in enumerate(masks_raw):
                    if is_rle_format:
                        # RLE dict format: {'size': [...], 'counts': [...]}
                        if not (isinstance(mask, dict) and 'size' in mask and 'counts' in mask):
                            raise ValueError(f'Mask {i} expected RLE dict format with "size" and "counts" keys.')
                        pred_masks[i] = mask
                    else:
                        # Numpy array format
                        if not isinstance(mask, np.ndarray):
                            raise ValueError(f'Mask {i} expected numpy array format.')
                        pred_masks[i] = mask.astype(np.uint8)
        
        pred_labels = [self.datalabel[_] for _ in pred_labels]

        # Create InstanceAnnotation objects
        annotations = []
        for i in range(len(pred_labels)):
            if pred_scores[i] >= cutoff:
                x1, y1, x2, y2 = pred_bboxes[i]
                bbox = cvtk.data.Bbox.from_xyxy(x1, y1, x2, y2, imsize=imsize) if imsize else None
                
                # Create segmentation from mask
                segm = None
                if pred_masks[i] is not None and imsize:
                    if isinstance(pred_masks[i], dict):
                        segm = cvtk.data.Segm.from_rle(pred_masks[i], imsize=imsize)
                    else:
                        segm = cvtk.data.Segm.from_mask(pred_masks[i])
                
                ann = cvtk.data.InstanceAnnotation(label=pred_labels[i], bbox=bbox, segm=segm, score=pred_scores[i])
                annotations.append(ann)
        
        return cvtk.data.ImageRecord(pathlib.Path(im_fpath), annotations=annotations, size=imsize)
        

    def __get_input_image_name_map(self):
        image_name_map = {}
        ann_file = getattr(self.cfg.test_evaluator, 'ann_file', None)
        if ann_file is None:
            return image_name_map

        try:
            with open(ann_file) as fh:
                cocodict = json.load(fh)
        except Exception:
            return image_name_map

        for image in cocodict.get('images', []):
            if 'id' in image and 'file_name' in image:
                image_name_map[image['id']] = image['file_name']
        return image_name_map


class SegmRunner(DetRunner):
    """Minimal instance-segmentation runner wrapper around DetRunner.

    This class keeps all training/testing/inference logic from DetRunner and
    only provides a task-specific alias for clearer API usage in segmentation
    workflows.
    """

    def __init__(self,
                 datalabel: cvtk.ml.data.DataLabel|str|list[str]|tuple[str],
                 cfg: str|dict,
                 weights: str|None=None,
                 workspace=None,
                 seed=None):
        super().__init__(datalabel=datalabel,
                         cfg=cfg,
                         weights=weights,
                         workspace=workspace,
                         seed=seed)
        self.task_type = 'segm'
    

