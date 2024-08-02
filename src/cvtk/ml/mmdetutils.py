import os
import re
import datetime
import shutil
import json
import random
import pathlib
import io
import base64
import copy
import filetype
import logging
import gzip
import tempfile
import pickle
import pandas as pd
import numpy as np
import importlib
import PIL
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import PIL.ImageFont
import skimage
import skimage.measure
import plotly.express as px
import plotly.subplots
import plotly.graph_objects as go
import pycocotools
import pycocotools.coco
import pycocotools.mask
import torch
import mim
import mmdet
import mmdet.utils
import mmdet.apis
import mmdet.models
import mmdet.datasets
import mmengine.config
import mmengine.registry
import mmengine.runner
import mmdet.evaluation
from cvtk import imread, Image, Annotation, ImageDeck, JsonComplexEncoder
from cvtk.format.coco import calc_stats
from cvtk.ml.data import DataLabel
from ._subutils import __del_docstring, __get_imports, __insert_imports, __extend_cvtk_imports


logger = logging.getLogger(__name__)


def DataPipeline(is_train: bool=False, with_bbox: bool=True, with_mask: bool=False) -> list[dict]:
    """Generate image preprocessing pipeline

    This class provides the basic image preprocessing pipeline used in MMDetection.

    Args:
        is_train: Whether the pipeline is for training. Default is False.
        with_bbox: Whether the dataset contains bounding boxes.
            Default is True for object detection with bounding boxes only.
        with_mask: Whether the dataset contains masks. Default is False.

    Attributes:
        train: The image preprocessing pipeline for training.
        valid: The image preprocessing pipeline for validation.
        test: The image preprocessing pipeline for testing. Same to the valid.
        inference: The image preprocessing pipeline for inference. Same to the valid.
    """
    if is_train:
        return [
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
        return [
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

    

def Dataset(datalabel: DataLabel, dataset: str|list[str]|dict|None, pipeline: list[dict]|None=None, repeat_dataset: bool=False):
    """Generate dataset configuration

    This function generates the dataset configuration for MMDetection.

    Args:
        datalabel: A DataLabel class object.
        dataset: A path to a COCO format file with extension '.json',
            a path to a directory containing images,
            a path to an image file, or a list of paths to image files.
            Note that, for training, validation, and test, the COCO format file is required.
        pipeline: The image preprocessing pipeline.
        repeat_dataset: Whether to repeat the dataset. Default is False.
            Use the repeated dataset for training will be faster in some architecture.
        
    """
    if dataset is None:
        return None
    elif isinstance(dataset, str) and dataset.endswith('.json'):
        dataset = dict(
            metainfo=dict(classes=datalabel.labels),
            type='CocoDataset',
            data_root='',
            data_prefix=dict(img=''),
            ann_file=os.path.abspath(dataset),
            pipeline=pipeline,
        )
        if repeat_dataset:
            dataset = dict(
                type='RepeatDataset',
                times=1,
                dataset=dataset
            )
    elif isinstance(dataset, (list, tuple)):
        dataset = dict(
            metainfo=dict(classes=datalabel.labels),
            type='CocoDataset',
            pipeline=pipeline,
            data_root=dataset
        )
    elif isinstance(dataset, str):
        dataset = dict(
            metainfo=dict(classes=datalabel.labels),
            type='CocoDataset',
            pipeline=pipeline,
            data_root=os.path.abspath(dataset)
        )
    elif isinstance(dataset, dict):
        pass
    else:
        raise TypeError(f'Invalid type: {type(dataset)}')
    return dataset


def DataLoader(dataset: dict|None=None, phase: str='inference', batch_size: int=4, num_workers: int=4) -> dict:
    """Generate dataloader configuration

    This function generates the dataloader configuration for MMDetection.

    Args:
        dataset: A dictionary of dataset configuration.
        phase: The purpose of DataLoader usage. It shold be specified as one
            'train', 'valie', 'test', and 'inference'.
        batch_size (int): Batch size.
        num_workers (int): Number of threads for data preprocessing and loading.
    """
    metrics = ['bbox']
    if dataset is not None:
        if 'pipeline' in dataset:
            for pp in dataset['pipeline']:
                if pp['type'] == 'LoadAnnotations':
                    if 'with_mask' in pp and pp['with_mask']:
                        metrics.append('segm')
    
    if phase == 'train':
        if dataset is None:
            raise ValueError('The dataset configuration is required for training, but got None.')
        else:
            return dict(
                dataset_type='CocoDataset',
                train_dataloader=dict(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    dataset=dataset,
                ),
                train_cfg = dict(
                    type='EpochBasedTrainLoop',
                    max_epochs=12,
                    val_interval=1,
                ),
            )
    elif phase == 'valid':
        if dataset is None:
            return dict(
                    val_dataloader=None,
                    val_cfg=None,
                    val_evaluator=None)
        else:
            return dict(
                    val_dataloader=dict(
                        _delete_=True,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        dataset=dataset,
                        drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=False)
                    ),
                    val_cfg = dict(
                        _delete_=True,
                        type='ValLoop'),
                    val_evaluator = dict(
                        _delete_=True,
                        type='CocoMetric',
                        ann_file=dataset['ann_file'],
                        metric=metrics,
                        backend_args=None
                    )
                )
    elif phase == 'test':
        if dataset is None:
            return dict(
                    test_dataloader=None,
                    test_cfg=None,
                    test_evaluator=None)
        else:
            return dict(
                    test_dataloader=dict(
                        _delete_=True,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        dataset=dataset,
                        drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=False)
                    ),
                    test_cfg = dict(
                        _delete_=True,
                        type='TestLoop'),
                    test_evaluator = dict(
                        _delete_=True,
                        type='CocoMetric',
                        ann_file=dataset['ann_file'],
                        metric=metrics,
                        backend_args=None
                    )
                )
    else: # other cases, e.g., inference
        return dict(test_dataloader=dict(
                    _delete_=True,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=False,
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    dataset=dataset))
    


class MMDETCORE():
    """A class for object detection and instance segmentation

    This class provides user-friendly APIs for object detection and instance segmentation
    using MMDetection.
    There are four main methods are implemented in this class:
    :func:`train <cvtk.ml.mmdetutils.MMDETCORE.train>`,
    :func:`test <cvtk.ml.mmdetutils.MMDETCORE.test>`,
    :func:`save <cvtk.ml.mmdetutils.MMDETCORE.save>`,
    :func:`inference <cvtk.ml.mmdetutils.MMDETCORE.inference>`.
    The :func:`train <cvtk.ml.mmdetutils.MMDETCORE.train>` method is used for training the model
    and perform validation and test if validation and test data are provided.
    The :func:`test <cvtk.ml.mmdetutils.MMDETCORE.test>` method is used for testing the model with test data.
    In general, the performance test is performed automatically after the training,
    but user can also run the test independently from the training process with
    the :func:`test <cvtk.ml.mmdetutils.MMDETCORE.test>` method.
    The :func:`save <cvtk.ml.mmdetutils.MMDETCORE.save>` method is used for saving the model weights,
    configuration (design of model architecture), training log (e.g., mAP and loss per epoch), and test results.
    The :func:`inference <cvtk.ml.mmdetutils.MMDETCORE.inference>` method is used for running inference
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
        >>> from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, MMDETCORE
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'stem'])
        >>> cfg = 'faster_rcnn_r50_fpn_1x_coco'
        >>> weights = None # download from MMDetection repository
        >>> workspace = '/path/to/workspace'
        >>>
        >>> model = MMDETCORE(datalabel, cfg, weights, workspace)
        >>> 
        >>> train = DataLoader('/path/to/train/coco.json', 'train')
        >>> model.train(train, epoch=10)
        >>> model.save('/path/to/model.pth')
    """
    def __init__(self,
                 datalabel: DataLabel|str|list[str]|tuple[str],
                 cfg: str|dict,
                 weights: str|None=None,
                 workspace=None,
                 seed=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datalabel = self.__init_datalabel(datalabel)
        self.cfg = self.__init_cfg(cfg, weights, seed)
        self.model = None
        self.tempd, self.workspace = self.__init_tempdir(workspace)
        self.mmdet_log_dpath = None
        self.test_stats = None
    

    def __del__(self):
        try:
            if self.tempd is not None:
                self.tempd.cleanup()
        except:
            logger.info(f'The temporary directory (`{self.workspace}`) created by cvtk '
                        f'cannot be removed automatically. Please remove it manually.')


    def __init_datalabel(self, datalabel):
        if isinstance(datalabel, DataLabel):
            pass
        elif isinstance(datalabel, str) or isinstance(datalabel, list) or isinstance(datalabel, tuple):
            datalabel = DataLabel(datalabel)
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
                    if isinstance(cfg[cfg_key][0], dict):
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


    def train(self,
              train: dict,
              valid: dict|None=None,
              test: dict|None=None,
              epoch: int=20,
              optimizer: dict|str|None=None,
              scheduler: dict|str|None=None):
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
        seed the :func:`test <cvtk.ml.mmdetutils.MMDETCORE.test>` method for more details.

        Args:
            train: A dictionary generated by DataLoader.
            valid: A dictionary generated by DataLoader or None.
            test: A dictionary generated by DataLoader or None.
            epoch: The number of epochs.
            optimizer: A dictionary of string indicating optimizer for training.
            scheduler: A dictionary of string indicating scheduler for training.

    Examples:
        >>> from cvtk.ml.data import DataLabel
        >>> from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, MMDETCORE
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'stem'])
        >>> cfg = 'faster_rcnn_r50_fpn_1x_coco'
        >>> weights = None # download from MMDetection repository
        >>> workspace = '/path/to/workspace'
        >>>
        >>> model = MMDETCORE(datalabel, cfg, weights, workspace)
        >>> 
        >>> train = DataLoader('/path/to/train/coco.json', 'train')
        >>> model.train(train, epoch=10)
        >>> model.save('/path/to/model.pth')
        >>>
        >>>
        >>> train = DataLoader('/path/to/train/coco.json', 'train')
        >>> valid = DataLoader('/path/to/valid/coco.json', 'valid')
        >>> test = DataLoader('/path/to/test/coco.json', 'test')
        >>> model.train(train, valid, test, epoch=10)
        >>> model.save('/path/to/model.pth')
        """
        self.cfg.device = self.device
        
        # training params
        self.__set_optimizer(optimizer)
        self.__set_scheduler(scheduler)
        
        # datasets
        self.cfg.merge_from_dict(train)
        
        self.cfg.train_cfg.max_epochs = epoch
        if valid is None:
            valid = DataLoader(None, 'valid')
        self.cfg.merge_from_dict(valid)
        self.cfg.merge_from_dict(DataLoader(None, 'test')) # test after training
        self.cfg.default_hooks.checkpoint.interval = 1000

        # training
        runner = mmengine.runner.Runner.from_cfg(self.cfg)
        self.mmdet_log_dpath = os.path.join(self.workspace, runner.timestamp)
        runner.train()
        del runner
        self.save(os.path.join(self.workspace, 'last_checkpoint.pth'))

        # test
        if test is not None:
            self.cfg.merge_from_dict(test)
            self.cfg.load_from = os.path.join(self.workspace, 'last_checkpoint.pth')
            self.test_stats = self.test(test)


    def __set_optimizer(self, optimizer):
        if optimizer is not None:
            if isinstance(optimizer, dict):
                opt_dict = optimizer
            elif isinstance(optimizer, str) and optimizer.replace(' ', '') != '':
                if optimizer[0] != '{' and optimizer[0:4] != 'dict':
                    optimizer = 'dict(' + optimizer + ')'
                opt_dict = eval(optimizer)
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
                scheduler_dict = eval(scheduler)
            self.cfg.param_scheduler = scheduler_dict


    def test(self, test:dict) -> dict:
        """Validate the model with test data
        
        This method is used to validate the model with test data.
        The test data should be COCO format file containing the annotations
        and converted to a dictionary withs :func:`DataLoader <cvtk.ml.mmdetutils.DataLoader>`.
        The predicted annotations of test data will be stored in the workspace
        with the names of :file:`test_outputs.pkl` in MMDetection format and
        :file:`test_outputs.coco.json` in COCO format.
        The performance metrics (e.g., mAP) will be returned as a dictionary.

        Args:
            test: A file path to COCO format file containing test data.    

        Examples:
        >>> from cvtk.ml.data import DataLabel
        >>> from cvtk.ml.mmdetutils import DataPipeline, Dataset, DataLoader, MMDETCORE
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'stem'])
        >>> cfg = 'faster_rcnn_r50_fpn_1x_coco'
        >>> weights = '/path/to/model.pth'
        >>>
        >>> model = MMDETCORE(datalabel, cfg, weights, workspace)
        >>> 
        >>> test = DataLoader('/path/to/test/coco.json', 'test')
        >>> metrics = model.test(test)
        >>> print(metrics)
        """
        self.cfg.merge_from_dict(test)
        runner = mmengine.runner.Runner.from_cfg(self.cfg)

        test_outputs = os.path.join(self.workspace, 'test_outputs.pkl')
        runner.test_evaluator.metrics.append(mmdet.evaluation.DumpDetResults(out_file_path=test_outputs))
        runner.test()

        with open(test_outputs, 'rb') as infh:
            pred_outputs = pickle.load(infh)

        cocodict = {'images': [], 'annotations': [], 'categories': []}
        for cate in self.datalabel.labels:
            cocodict['categories'].append({
                'id': self.datalabel[cate],
                'name': cate
            })

        annid = 0
        for po in pred_outputs:
            cocodict['images'].append({
                'id': po['img_id'],
                'file_name': po['img_path'] if 'img_path' in po else None,
                'width': po['ori_shape'][1] if 'ori_shape' in po else None,
                'height': po['ori_shape'][0] if 'ori_shape' in po else None
            })

            if 'pred_instances' in po:
                imanns = self.__format_mmdet_output(po['img_path'], po['pred_instances'], cutoff=0)
                for j, imann in enumerate(imanns.annotations):
                    annid += 1
                    cocodict['annotations'].append({
                        'id': annid,
                        'image_id': cocodict['images'][-1]['id'],
                        'category_id': self.datalabel[imann['label']],
                        'score': imann['score'],
                        'bbox': self.__xyxy2xywh(imann['bbox']),
                        'area': imann['area'],
                        'iscrowd': 0
                    })
                    if 'mask' in imann and imann['mask'] is not None:
                        cocodict['annotations'][-1]['segmentation'] = {
                            'size': po['pred_instances']['masks'][j]['size'],
                            'counts': po['pred_instances']['masks'][j]['counts'].decode(),
                        }

        with open(os.path.splitext(test_outputs)[0] + '.coco.json', 'w') as oufh:
            json.dump(cocodict, oufh, cls=JsonComplexEncoder, indent=4)

        iou_type = 'bbox'
        for pp in self.cfg.test_dataloader.dataset.pipeline:
            if pp['type'] == 'LoadAnnotations':
                if 'with_mask' in pp and pp['with_mask']:
                    iou_type = 'segm'

        self.test_stats = calc_stats(self.cfg.test_evaluator.ann_file,
                                     os.path.splitext(test_outputs)[0] + '.coco.json',
                                     iouType=iou_type)
        return self.test_stats
    
    
    def __xyxy2xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return [x, y, w, h]
    

    def save(self, output: str):
        """Save the model

        Save the model. If training metrics and test results,
        usually generated from training process,
        are exists, they will be save in the same name of weights but
        with the different suffixes.

        Args:
            output: A path to store the model weights and configuration.
        
        """
        if not output.endswith('.pth'):
            output + '.pth'
        if not os.path.exists(os.path.dirname(output)):
            if os.path.dirname(output) != '':
                os.makedirs(os.path.dirname(output))

        with open(os.path.join(self.workspace, 'last_checkpoint')) as chkf:
            shutil.copy2(chkf.readline().strip(), output)
        
        cfg_fpath = os.path.splitext(output)[0] + '.py'
        self.cfg.dump(cfg_fpath)

        self.__write_trainlog(os.path.splitext(output)[0] + '.train_stats')

        if self.test_stats is not None:
            with open(os.path.join(os.path.splitext(output)[0] + '.test_stats.json'), 'w') as outfh:
                json.dump(self.test_stats, outfh, indent=4, ensure_ascii=False)


    def __write_trainlog(self, output_prefix=None):
        train_log = []
        valid_log = []

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


    def inference(self, data: dict|str|list[str], cutoff: float=0) -> list[Image]:
        """Inference

        perform inference.
        
        Args:
            data: A path to a directory containing images,
                a path to an image file, or a list of paths to image files.
            score_cutoff (float): The score cutoff for inference. Default is 0.5.

        Examples:
            >>> test_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']
            >>> outputs = model.inference(sample_images)
            >>> for output in outputs:
            >>>     bbox_img_fpath = os.path.splitext(output.file_path)[0] + '.bbox.png'
            >>>     output.draw('bbox+segm', output=bbox_img_fpath)
            >>>
            >>> coco_json = model.inference(sample_images)
            >>> with open('inference_results.json', 'w') as oufh:
            >>>     json.dump(coco_json, outfh)
        """
        input_images = []
        if isinstance(data, dict):
            # test dataloader defined by mmdet
            self.cfg.merge_from_dict(data)
            data_dpath = self.cfg.test_dataloader.dataset.data_root
            if data_dpath == '':
                if self.cfg.test_dataloader.dataset.type == 'RepeatDataset':
                    data_dpath = self.cfg.test_dataloader.dataset.dataset.ann_file
                else:
                    data_dpath = self.cfg.test_dataloader.dataset.ann_file
            input_images = self.__load_images(data_dpath)
            self.cfg.test_dataloader.dataset.data_root = ''
        else:
            input_images = self.__load_images(data)

        if self.model is None:
            self.model = mmdet.apis.init_detector(self.cfg, self.cfg.load_from, device=self.device)
        pred_outputs = mmdet.apis.inference_detector(self.model, input_images)
        
        # format
        outputs_fmt = []
        for target_image, output in zip(input_images, pred_outputs):
            outputs_fmt.append(self.__format_mmdet_output(target_image, output.pred_instances, cutoff))

        return outputs_fmt
    
    
    def __load_images(self, dataset):
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
                        for im in cocofh['images']:
                            x.append(im['file_name'])
                    else:
                        x = []
                        for line in trainfh:
                            words = line.rstrip().split('\t')
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
            raise ValueError('No image files found in the given dataset ({dataset}).')

        return x
    
    
    def __format_mmdet_output(self, im_fpath, pred_instances, cutoff):
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
        
        if 'masks' in pred_instances:
            pred_masks = []
            pred_masks_ = None
            if isinstance(pred_instances, dict):
                pred_masks_ = pred_instances['masks']
            else:
                pred_masks_ = pred_instances.masks.detach().cpu().numpy()
            for pred_mask_ in pred_masks_:
                if isinstance(pred_mask_, dict):
                    if 'size' in pred_mask_ and 'counts' in pred_mask_:
                        pred_masks.append(pycocotools.mask.decode(pred_mask_))
                    else:
                        raise ValueError('The mask is expected to have "size" and "counts" when dict is given.')
                elif isinstance(pred_mask_, np.ndarray):
                    pred_masks.append(pred_mask_.astype(np.uint8))
                else:
                    raise ValueError('The mask is expected to be a dictionary or numpy array.')
        else:
            pred_masks = [None] * len(pred_bboxes)
        
        pred_labels = [self.datalabel[_] for _ in pred_labels]

        pred_labels_ = []
        pred_bboxes_ = []
        pred_masks_ = []
        pred_scores_ = []
        for i in range(len(pred_labels)):
            if pred_scores[i] >= cutoff:
                pred_labels_.append(pred_labels[i])
                pred_bboxes_.append(pred_bboxes[i])
                pred_masks_.append(pred_masks[i])
                pred_scores_.append(pred_scores[i])

        imann = Annotation(pred_labels_, pred_bboxes_, pred_masks_, pred_scores_)
        return Image(im_fpath, annotations=imann)
        


def plot_trainlog(train_log, y=None, output=None, title='Training Statistics', mode='lines', width=600, height=800, scale=1.9):
    """Plot train log.

    Plot train log. `train_log` format should be:

    ::

        epoch   lr      data_time       loss    loss_rpn_cls    loss_rpn_bbox   loss_cls        acc     loss_bbox       time    memory
        1       2e-05   0.6311202049255371      2.71342134475708        0.3602621555328369      0.05165386199951172     1.4231624603271484      16.58203125     0.8783428072929382      2.2113988399505615  15742.0
        2       6.004008016032065e-05   0.4661067724227905      2.708804130554199       0.3569621592760086      0.05149035155773163     1.4229193925857544      17.0703125      0.877432256937027   1.7055927515029907      15974.0
        3       0.0001000801603206413   0.4101251761118571      2.6866095860799155      0.3382184902826945      0.05082566291093826     1.4187644720077515      19.62890625     0.8788009683291117  1.5375737349192302      15974.0
        4       0.00014012024048096195  0.3806012272834778      2.6515525579452515      0.3062228001654148      0.04974065348505974     1.411013811826706       23.69140625     0.8845753073692322  1.4551105499267578      15974.0
        5       0.00018016032064128258  0.36423935890197756     2.603787565231323       0.2676710680127144      0.048102487623691556    1.3975130558013915      33.4375 0.8905009746551513 1.403717279434204        15974.0

    `valid_log` should be:

    ::

        coco/bbox_mAP   coco/bbox_mAP_50        coco/bbox_mAP_75        coco/bbox_mAP_s coco/bbox_mAP_m coco/bbox_mAP_l data_time       time    step
        0.001   0.003   0.0     -1.0    -1.0    0.001   0.6635150909423828      1.0537631511688232      1
        0.001   0.004   0.0     -1.0    -1.0    0.001   0.4849787950515747      0.861297607421875       2
        0.001   0.006   0.0     -1.0    -1.0    0.003   0.30100834369659424     0.661655068397522       3
        0.001   0.006   0.0     -1.0    -1.0    0.004   0.2974175214767456      0.6560839414596558      4
        0.001   0.007   0.0     -1.0    -1.0    0.003   0.29656195640563965     0.6557941436767578      5
        0.001   0.005   0.0     -1.0    -1.0    0.001   0.29982125759124756     0.6613177061080933      6

    
    """
    # data preparation
    train_log = pd.read_csv(train_log, sep='\t', header=0, comment='#')
        
    # coordinates
    x = None
    if 'epoch' in train_log.columns.values.tolist():
        x = 'epoch'
        if y is None:
            y = ['loss', 'loss_cls', 'loss_bbox', 'acc']
    else:
        x = 'step'
        if y is None:
            y = ['coco/bbox_mAP', 'coco/bbox_mAP_50']

    # plots
    cols = px.colors.qualitative.Plotly
    fig = plotly.subplots.make_subplots(rows=len(y), cols=1)
    for y_ in y:
        fig.add_trace(
            go.Scatter(x=train_log[x], y=train_log[y_],
                       mode=mode,
                       name=y_,
                       line=dict(color='#333333'),
                       showlegend=False),
            row=y.index(y_) + 1, col=1)
        fig.update_yaxes(title_text=y_, row=y.index(y_) + 1, col=1)
    
    fig.update_layout(title_text=title, template='ggplot2')
    fig.update_xaxes(title_text=x)

    if output is not None:
        fig.write_image(output, width=width, height=height, scale=scale)
    else:
        fig.show()
    return fig


def __generate_source(script_fpath, task, vanilla=False):
    if not script_fpath.endswith('.py'):
        script_fpath += '.py'

    tmpl = ''
    with open(importlib.resources.files('cvtk').joinpath('tmpl/_mmdet.py'), 'r') as infh:
        tmpl = infh.readlines()

    if vanilla is True:
        cvtk_modules = [
            {'cvtk': [JsonComplexEncoder, Annotation, Image, ImageDeck, imread]},
            {'cvtk.format.coco': [calc_stats]},
            {'cvtk.ml.data': [DataLabel]},
            {'cvtk.ml.mmdetutils': [DataPipeline, Dataset, DataLoader, MMDETCORE, plot_trainlog]}
        ]
        tmpl = __insert_imports(tmpl, __get_imports(__file__))
        tmpl = __extend_cvtk_imports(tmpl, cvtk_modules)

    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__SCRIPTNAME__', os.path.basename(script_fpath))
    if task.lower()[:3] == 'det':
        tmpl = tmpl.replace('__TASKARCH__', 'faster-rcnn_r101_fpn_1x_coco')
        tmpl = tmpl.replace('__SAMPLEDATA__', 'bbox.json')
    else:
        tmpl = tmpl.replace('__TASKARCH__', 'mask-rcnn_r101_fpn_1x_coco')
        tmpl = tmpl.replace('__SAMPLEDATA__', 'segm.json')
        tmpl = tmpl.replace('with_mask=False', 'with_mask=True')
    tmpl = __del_docstring(tmpl)

    with open(script_fpath, 'w') as fh:
        fh.write(tmpl)

