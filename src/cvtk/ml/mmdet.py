import os
import shutil
import json
import filetype
import logging
import gzip
import tempfile
import random
import pandas as pd
import PIL
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
from .data import DataClass
from ._base import __generate_cvtk_imports_string, __del_docstring
try:
    import skimage
    import skimage.measure
    import pycocotools.coco
    import torch
    import mim
    import mmcv
    import mmdet
    import mmcv.ops
    import mmdet.utils
    import mmdet.apis
    import mmdet.models
    import mmdet.datasets
    import mmengine.config
    import mmengine.registry
    import mmengine.runner
    
except ImportError as e:
    raise ImportError('Unable to import mmdet related packages. '
                      'Install mmcv, mmdet, and the related packages to enable this feature.') from e

logger = logging.getLogger(__name__)


def DataPipeline(is_train=False, with_bbox=True, with_mask=False):
    """Generate image preprocessing pipeline

    This class provides the basic image preprocessing pipeline used in MMDetection.

    Args:
        is_train (bool): Whether the pipeline is for training. Default is False.
        with_bbox (bool): Whether the dataset contains bounding boxes.
            Default is True for object detection with bounding boxes only.
        with_mask (bool): Whether the dataset contains masks. Default is False.

    Attributes:
        train (list): The image preprocessing pipeline for training.
        valid (list): The image preprocessing pipeline for validation.
        test (list): The image preprocessing pipeline for testing. Same to the valid.
        inference (list): The image preprocessing pipeline for inference. Same to the valid.
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



@mmdet.registry.DATASETS.register_module()
class DataSetCfg():
    """Generate dataset configuration

    This class generates the dataset configuration for MMDetection.
    Currently only support COCO dataset.

    Args:
        dataclass (DataClass): The class
    """

    METAINFO = {
        'classes': None,
        'palette': None
    }

    def load_data_list(self):
        self.coco =  pycocotools.coco.COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.metainfo['classes'])
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

        img_ids = self.coco.getImgIds()
        data_list = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_info['id'] = img_id
            
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)

            data_info = self.parse_data_info(img_info, ann_info)
            data_list.append(data_info)

        return data_list
    

    def parse_data_info(self, img_info, ann_info):
        data_info = {}

        data_info = {
            'img_id': img_info['id'],
            'img_path': os.path.join(self.data_prefix['img'], img_info['file_name']),
            'seg_map_path': None,
            'height': img_info['height'],
            'width': img_info['width']
        }

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True
        
        instances = []
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            instance = {}
            instance['bbox'] = [x1, y1, x1 + w, y1 + h]
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            
            instances.append(instance)

        data_info['instances'] = instances
        return data_info


    


def Dataset(dataclass, dataset=None, pipeline=None, repeat_dataset=False):
    """Generate dataset configuration

    This function generates the dataset configuration for MMDetection.

    Args:
        dataclass (DataClass): The class
        dataset (list|str): The path to the dataset.
        pipeline (list): The image preprocessing pipeline.
        repeat_dataset (bool): Whether to repeat the dataset. Default is False.
            Use the repeated dataset for training will be faster in some architecture.
        
    """
    if isinstance(dataset, str) and dataset.endswith('.json'):
        dataset = dict(
            metainfo=dict(classes=dataclass.classes),
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
    else:
        dataset = dict(
            metainfo=dict(classes=dataclass.classes),
            type='CocoDataset',
            pipeline=pipeline,
            data_root=os.path.abspath(dataset)
        )
    return dataset


def DataLoader(dataset, phase='inference', batch_size=4, num_workers=4):
    """Generate dataloader configuration

    This function generates the dataloader configuration for MMDetection.

    Args:
        dataset (dict): The dataset configuration.
        phase (str): The phase of the dataloader. 'train', 'valid', 'test', 'inference' are supported.
            For prediction, use 'inference'.
        batch_size (int): The batch size for training, validation, and testing.
        num_workers (int): The number of CPU cores for data loading.
    """
    metrics = ['bbox']
    if 'pipeline' in dataset:
        for pp in dataset['pipeline']:
            if pp['type'] == 'LoadAnnotations':
                if 'with_mask' in pp and pp['with_mask']:
                    metrics.append('segm')
    
    if phase == 'train':
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
        return dict(
                val_dataloader=dict(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    dataset=dataset,
                    drop_last=False,
                    sampler=dict(type='DefaultSampler', shuffle=False)
                ),
                val_cfg = dict(type='ValLoop'),
                val_evaluator = dict(
                    type='CocoMetric',
                    ann_file=dataset['ann_file'],
                    metric=metrics,
                    backend_args=None
                )
            )
    elif phase == 'test':
        return dict(
                test_dataloader=dict(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    dataset=dataset,
                    drop_last=False,
                    sampler=dict(type='DefaultSampler', shuffle=False)
                ),
                test_cfg = dict(type='TestLoop'),
                test_evaluator = dict(
                    type='CocoMetric',
                    ann_file=dataset['ann_file'],
                    metric=metrics,
                    backend_args=None
                )
            )
    else:
        return dict(test_dataloader=dict(
                    _delete_=True,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=False,
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    dataset=dataset))
    


class MMDETCORE():
    """A class for MMDetection core functions


    Run `mim search mmdet --model "faster r-cnn"` to set the pre-defined configuration for `cfg`.

    Args:
        dataclass (DataClass): The class for the dataset.
        cfg (str): The path to the configuration file.
        weights (str): The path to the model weights.
        workspace (str): The path to the workspace. Default is None.


    """
    def __init__(self,
                 dataclass,
                 cfg,
                 weights=None,
                 workspace=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataclass = self.__init_dataclass(dataclass)
        self.cfg = self.__init_cfg(cfg, weights)
        self.tempd = self.__init_tempdir(workspace)
        self.mmdet_log_dpath = None

    
    def __del__(self):
        try:
            if self.tempd is not None:
                self.tempd.cleanup()
        except:
            logger.info(f'The temporary directory (`{self.cfg.work_dir}`) created by cvtk '
                        f'cannot be removed automatically. Please remove it manually.')


    def __init_dataclass(self, dataclass):
        if isinstance(dataclass, DataClass):
            pass
        if isinstance(dataclass, str) or isinstance(dataclass, list) or isinstance(dataclass, tuple):
            dataclass = DataClass(dataclass)
        elif not isinstance(dataclass, DataClass):
            raise TypeError('Invalid type: {}'.format(type(dataclass)))
        return dataclass


    def __init_cfg(self, cfg, weights):
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
            raise TypeError('Invalid type: {}'.format(type(cfg)))
        
        if weights is None:
            if chk is not None:
                cfg.load_from = chk
        else:
            if os.path.exists(weights):
                cfg.load_from = weights
            else:
                raise FileNotFoundError('The model weights file does not exist.')

        cfg.launcher = 'none'
        cfg.resume = False
        cfg = self.__set_classes(cfg, self.dataclass.classes)
        return  cfg


    def __set_classes(self, cfg, class_labels):
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
        return tempd


    def train(self,
              train,
              valid=None,
              test=None,
              epoch=20,
              optimizer=None,
              scheduler=None):
        """Train detection model

        Train model.

        Args:
            train (DataLoader): The path to the training dataset.
            valid (DataLoader): The path to the validation dataset. Default is None.
            test (DataLoader): The path to the test dataset. Default is None.
            epoch (int): The number of epochs. Default is 20.
            optimizer (dict): The optimizer configuration. Default is None.
            scheduler (dict): The scheduler configuration. Default is None.
        
        """
        self.cfg.device = self.device
        
        # training params
        if optimizer is not None:
            self.cfg.optim_wrapper = optimizer
        if scheduler is not None:
            self.cfg.param_scheduler = scheduler
        
        # datasets
        self.cfg.merge_from_dict(train)
        
        self.cfg.train_cfg.max_epochs = epoch
        if valid is not None:
            self.cfg.merge_from_dict(valid)
        if test is not None:
            self.cfg.merge_from_dict(test)
        self.cfg.default_hooks.checkpoint.interval = 10

        # training
        runner = mmengine.runner.Runner.from_cfg(self.cfg)
        self.mmdet_log_dpath = os.path.join(self.cfg.work_dir, runner.timestamp)
        runner.train()


    def save(self, output):
        """Save the model

        Save the model.

        Args:
            output (str): The path to the output file.
        
        """
        if not output.endswith('.pth'):
            output + '.pth'
        if not os.path.exists(os.path.dirname(output)):
            if os.path.dirname(output) != '':
                os.makedirs(os.path.dirname(output))

        with open(os.path.join(self.cfg.work_dir, 'last_checkpoint')) as chkf:
            shutil.copy2(chkf.readline().strip(), output)
        
        cfg_fpath = os.path.splitext(output)[0] + '.py'
        self.cfg.dump(cfg_fpath)

        self.__write_trainlog(os.path.splitext(output)[0] + '.train_stats')



    def __write_trainlog(self, output_prefix=None):
        train_log = []
        valid_log = []

        log_fpath = os.path.join(self.mmdet_log_dpath, 'vis_data', 'scalars.json')
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



    def inference(self, dataloader, score_cutoff=0.5):
        """Inference

        perform inference.
        
        Args:
            dataloader (DataLoader): The dataloader configuration.
            score_cutoff (float): The score cutoff for inference. Default is 0.5.
        """
        self.cfg.merge_from_dict(dataloader)
        model = mmdet.apis.init_detector(self.cfg, self.cfg.load_from, device=self.device)

        input_images = self.__load_images(self.cfg.test_dataloader.dataset.data_root)
        self.cfg.test_dataloader.dataset.data_root = ''
 
        pred_outputs = mmdet.apis.inference_detector(model, input_images)
        
        # format
        outputs_fmt = []
        for target_image, output in zip(input_images, pred_outputs):
            outputs_fmt.append(self.__format_output(target_image, output, score_cutoff))
        
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
        return x
    
    
    def __format_output(self, input, output, score_cutoff):
        if 'bboxes' in output.pred_instances:
            pred_bboxes = output.pred_instances.bboxes.detach().cpu().numpy().tolist()
            pred_labels = output.pred_instances.labels.detach().cpu().numpy().tolist()
            pred_scores = output.pred_instances.scores.detach().cpu().numpy().tolist()
        else:
            pred_bboxes = []
            pred_labels = []
            pred_scores = []
        
        if 'masks' in output.pred_instances:
            pred_masks = output.pred_instances.masks.detach().cpu().numpy()
        else:
            pred_masks = [None] * len(pred_bboxes)
        
        regions = []
        for pred_bbox, pred_label, pred_score, pred_mask in zip(
                pred_bboxes, pred_labels, pred_scores, pred_masks):
            if pred_score > score_cutoff:
                region = {
                    'class': self.dataclass[pred_label],
                    'score': pred_score,
                    'bbox': pred_bbox
                }
                if pred_mask is not None:
                    sg = skimage.measure.find_contours(pred_mask, 0.5)
                    region['contour'] = sg[0][:, [1, 0]].tolist()
                regions.append(region)
        
        return {'image': input, 'annotations': regions}



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
    import pandas as pd
    import plotly.express as px
    import plotly.subplots
    import plotly.graph_objects as go

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



def draw_outlines(image_fpath, output_fpath, outlines, col=None):
    """Draw bbox and contour

    Args:
        image_fpath (str): The path to the image.
        output_fpath (str): The path to the output image.
        outlines (list): The list of outlines. Each element should be a dictionary with the following keys
            - bbox: The bounding box coordinates. (required)
            - class: The class label. (optional)
            - contour: The contour coordinates. (optional)
        col (dict): The color dictionary. Default is None.
    """

    font = PIL.ImageFont.load_default(10)

    im = PIL.Image.open(image_fpath)
    im = PIL.ImageOps.exif_transpose(im)
    imdr = PIL.ImageDraw.Draw(im)

    if col is None:
        col = {}
    
    for outline in outlines:
        if 'class' in outline:
            if outline['class'] not in col:
                col[outline['class']] = (random.randint(0, 255),
                                         random.randint(0, 255),
                                         random.randint(0, 255))
        else:
            col['___UNDEF___'] = (random.randint(0, 255),
                                  random.randint(0, 255),
                                  random.randint(0, 255))
        
        x1, y1, x2, y2 = outline['bbox']
        cl = outline['class'] if 'class' in outline else '___UNDEF___'
        
        imdr.rectangle([(x1, y1), (x2, y2)], outline = col[cl], width=5)
        if 'contour' in outline:
            xy = [tuple(c) for c in outline['contour']]
            imdr.line(xy, fill = col[cl], width=5)
        if 'class' in outline:
            imdr.text((x1, y1), cl, font=font)
    im.save(output_fpath)



def __generate_source(script_fpath, task, module='cvtk'):
    if not script_fpath.endswith('.py'):
        script_fpath += '.py'
    
    cvtk_modules = [
        {'cvtk.ml.data': [DataClass]},
        {'cvtk.ml.mmdet': [DataPipeline, Dataset, DataLoader, MMDETCORE, plot_trainlog, draw_outlines]}
    ]
    function_imports = __generate_cvtk_imports_string(cvtk_modules, import_from=module)

    task_code = 'with_bbox=True'
    sample_data = 'bbox.json'
    task_arch = 'faster-rcnn_r101_fpn_1x_coco'
    if task.lower()[:3] == 'seg':
        task_code += ', with_mask=True'
        sample_data = 'segm.json'
        task_arch = 'mask-rcnn_r101_fpn_1x_coco'


    tmpl = f'''import os
import argparse
import shutil
import json
import filetype
import logging
import gzip
import tempfile
import random
import inspect
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import PIL.ImageFont
import skimage
import skimage.measure
import pycocotools.coco
import torch
import mim
import mmcv
import mmdet
import mmcv.ops
import mmdet.utils
import mmdet.apis
import mmdet.models
import mmdet.datasets
import mmengine.config
import mmengine.registry
import mmengine.runner
import logging
logger = logging.getLogger(__name__)

{function_imports}


def train(dataclass, train, valid, test, output_weights, batch_size=4, num_workers=8, epoch=10):
    temp_dpath = os.path.splitext(output_weights)[0]

    # class labels
    dataclass = DataClass(dataclass)

    # model setup
    model = MMDETCORE(dataclass, '{task_arch}', None, workspace=temp_dpath)

    # datasets
    train = DataLoader(
                Dataset(dataclass, train, DataPipeline(is_train=True, {task_code})),
                phase='train', batch_size=batch_size, num_workers=num_workers)
    valid = DataLoader(
                Dataset(dataclass, valid, DataPipeline(is_train=False, {task_code})),
                phase='valid', batch_size=batch_size, num_workers=num_workers)
    test = DataLoader(
                Dataset(dataclass, test, DataPipeline(is_train=False, {task_code})),
                phase='test', batch_size=batch_size, num_workers=num_workers)
    
    model.train(train, valid, test, epoch=epoch)
    model.save(output_weights)

    # plot train log
    plot_trainlog(os.path.splitext(output_weights)[0] + '.train_stats.train.txt',
                  output=os.path.splitext(output_weights)[0] + '.train_stats.train.png')
    plot_trainlog(os.path.splitext(output_weights)[0] + '.train_stats.valid.txt',
                  output=os.path.splitext(output_weights)[0] + '.train_stats.valid.png')


def inference(dataclass, data, model_weights, output, batch_size=4, num_workers=8):
    temp_dpath = os.path.splitext(output)[0]

    # class labels
    dataclass = DataClass(dataclass)

    # model setup
    model = MMDETCORE(dataclass, os.path.splitext(model_weights)[0] + '.py', model_weights, workspace=temp_dpath)

    # datasets
    data = DataLoader(
                Dataset(dataclass, data, DataPipeline()),
                phase='inference', batch_size=batch_size, num_workers=num_workers)
    
    pred_outputs = model.inference(data)

    with open(output, 'w') as fh:
        json.dump({{'data': pred_outputs}}, fh, ensure_ascii=False, indent=4)
    
    for pred_output in pred_outputs:
        draw_outlines(pred_output['image'],
                      os.path.join(temp_dpath, os.path.basename(pred_output['image'])),
                      pred_output['annotations'])


def _train(args):
    train(args.dataclass, args.train, args.valid, args.test, args.output_weights)

    
def _inference(args):
    inference(args.dataclass, args.data, args.model_weights, args.output)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataclass', type=str, required=True)
    parser_train.add_argument('--train', type=str, required=True)
    parser_train.add_argument('--valid', type=str, required=False)
    parser_train.add_argument('--test', type=str, required=False)
    parser_train.add_argument('--output_weights', type=str, required=True)
    parser_train.set_defaults(func=_train)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--dataclass', type=str, required=True)
    parser_inference.add_argument('--data', type=str, required=True)
    parser_inference.add_argument('--model_weights', type=str, required=True)
    parser_inference.add_argument('--output', type=str, required=False)
    parser_inference.set_defaults(func=_inference)

    args = parser.parse_args()
    args.func(args)
    
'''
    
    tmpl = __del_docstring(tmpl)
    tmpl += f'''
"""
Example Usage:


python __scriptname__ train \\
    --dataclass ./data/strawberry/class.txt \\
    --train ./data/strawberry/train/{sample_data}.json \\
    --valid ./data/strawberry/valid/{sample_data}.json \\
    --test ./data/strawberry/test/{sample_data}.json \\
    --output_weights ./output/strawberry.pth

    
python __scriptname__ inference \\
    --dataclass ./data/strawberry/class.txt \\
    --data ./data/strawberry/test/images \\
    --model_weights ./output/arts.pth \\
    --output ./output/pred_results

"""
'''

    tmpl = tmpl.replace('__scriptname__', os.path.basename(script_fpath))
   
    with open(script_fpath, 'w') as fh:
        fh.write(tmpl)

