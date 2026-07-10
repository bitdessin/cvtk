import math
import os
import copy
import json
import random
from typing import Literal
import numpy as np
import PIL
import PIL.Image
import pycocotools
import pycocotools.coco
import pycocotools.cocoeval

from .. import utils as cvtk_utils
from . import _im as cvtk_data


_METRICS_LABELS = ['AP@[0.50:0.95|all|100]',
                   'AP@[0.50|all|1000]',
                   'AP@[0.75|all|1000]',
                   'AP@[0.50:0.95|small|1000]',
                   'AP@[0.50:0.95|medium|1000]',
                   'AP@[0.50:0.95|large|1000]',
                   'AR@[0.50:0.95|all|100]',
                   'AR@[0.50:0.95|all|300]',
                   'AR@[0.50:0.95|all|1000]',
                   'AR@[0.50:0.95|small|1000]',
                   'AR@[0.50:0.95|medium|1000]',
                   'AR@[0.50:0.95|large|1000]']


def _load_coco(input, image_root=None):
    def _get_path(file_name, image_root=None):
        if image_root is None or file_name is None:
            return file_name
        return os.path.join(image_root, file_name)
    
    if isinstance(input, str):
        with open(input, 'r') as f:
            cocodata = json.load(f)
    else:
        cocodata = copy.deepcopy(input)

    if image_root is not None:
        for image in cocodata.get('images', []):
            if 'file_name' in image:
                image['file_name'] = _get_path(image['file_name'], image_root)

    return cocodata


def crop(
    input: str|dict,
    image_root: str|None=None,
    output: str|None=None,
) -> None:
    """Crop objects from images based on COCO annotations.

    Extracts individual annotated objects from images and saves them as separate cropped images.
    Each crop is named using the image filename, category ID, and bounding box coordinates.

    Args:
        input (str|dict): COCO annotation data. Can be a file path to JSON or a dict object.
        image_root (str|None): Base directory for image paths. If None, uses paths as stored in annotations. Default is None.
        output (str|None): Directory to save cropped object images. Directory created if needed.

    Returns:
        None. Cropped images saved to output directory.

    Raises:
        ValueError: If output directory not specified or image not found for annotation.

    Examples:
        >>> crop('annotations.json', output='cropped_objects/')
        >>> crop('annotations.json', image_root='/data/images', output='crops/')
    """
    if output is None:
        raise ValueError('`output` must be specified to save the cropped objects.')
    if not os.path.exists(output):
        os.makedirs(output)
    
    cocodata = _load_coco(input, image_root=image_root)

    for ann in cocodata['annotations']:
        im_path = None
        for _ in cocodata['images']:
            if _['id'] == ann['image_id']:               
                im_path =_['file_name']
                break
            
        if im_path is None:
            raise ValueError(f'Image with ID {ann["image_id"]} not found in COCO annotation data.')
        
        im = PIL.Image.open(im_path)
        x1, y1, x2, y2 = cvtk_data.Bbox.xywh2xyxy(ann['bbox'])
        x1 = max(0, math.floor(x1))
        y1 = max(0, math.floor(y1))
        x2 = min(math.ceil(x2), im.size[0])
        y2 = min(math.ceil(y2), im.size[1])
        
        im_crop = im.crop((x1, y1, x2, y2))
        fname, fext = os.path.splitext(os.path.basename(im_path))
        im_crop.save(
            os.path.join(output, '{}_{}_{}{}'.format(
                fname,
                ann['category_id'],
                '-'.join([str(int(i)) for i in (x1, y1, x2, y2)]),
                fext)))


def combine(
    input: str|dict|list[str]|list[dict]|tuple[str]|tuple[dict],
    image_root: str|None=None,
    output: str|None=None,
    ensure_ascii: bool=False,
    indent: int|None=4
) -> dict:
    """Merge multiple COCO annotation files into one.

    Combines multiple COCO datasets by merging all images, annotations, and categories.
    All IDs are re-indexed sequentially to avoid conflicts. Duplicate categories are deduplicated by name.

    Args:
        input (str|dict|list|tuple): Single or multiple COCO sources. Can be file path(s) or dict object(s).
        image_root (str|None): Base directory for image paths. If None, uses paths from annotations. Default is None.
        output (str|None): File path to save merged COCO annotation. If None, only returns data without saving. Default is None.
        ensure_ascii (bool): If True, escapes non-ASCII characters in JSON output. Default is False.
        indent (int|None): JSON indentation level. If None, compact output. Default is 4.

    Returns:
        dict: Merged COCO annotation data with re-indexed IDs and deduplicated categories.

    Examples:
        >>> combined = combine(['ann1.json', 'ann2.json'], output='merged.json')
        >>> combined = combine([coco_dict1, coco_dict2], image_root='/data/images')
    """
    merged_coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    image_id = 1
    category_id = 1
    annotation_id = 1
    
    for input_file in cvtk_utils.as_list(input):
        image_idmap = {}
        category_idmap = {}
        cocodata = _load_coco(input_file, image_root=image_root)
            
        for category in cocodata['categories']:
            if category['name'] not in [c['name'] for c in merged_coco['categories']]:
                category_idmap[category['id']] = category_id
                category['id'] = category_id
                merged_coco['categories'].append(category)
                category_id += 1
            else:
                category_idmap[category['id']] = [c['id'] for c in merged_coco['categories'] if c['name'] == category['name']][0]
            
        for image in cocodata['images']:
            image_idmap[image['id']] = image_id
            image['id'] = image_id
            merged_coco['images'].append(image)
            image_id += 1
            
        for annotation in cocodata['annotations']:
            annotation['id'] = annotation_id
            annotation['image_id'] = image_idmap[annotation['image_id']]
            annotation['category_id'] = category_idmap[annotation['category_id']]
            merged_coco['annotations'].append(annotation)
            annotation_id += 1
    
    if output is not None:
        cvtk_utils.save_json(merged_coco, output, indent=indent, ensure_ascii=ensure_ascii)
    return merged_coco


def split(
    input: str|dict,
    image_root: str|None=None,
    ratios: list[float]|tuple[float]=[0.8, 0.1, 0.1],
    shuffle: bool=True,
    random_seed: int|None=None,
    output: str|None=None,
    ensure_ascii=False,
    indent=4
) -> list[dict]:
    """Split COCO annotation data into train/validation/test subsets.

    Partitions images into multiple subsets with specified ratios. Annotations follow images.
    Categories are shared across all subsets. Optionally shuffles before splitting for randomization.

    Args:
        input (str|dict): COCO annotation data as file path or dict object.
        image_root (str|None): Base directory for image paths. If None, uses paths from annotations. Default is None.
        ratios (list|tuple): Subset size ratios. Sum must equal 1.0. Default is [0.8, 0.1, 0.1].
        shuffle (bool): If True, randomize image order before splitting. Default is True.
        random_seed (int|None): Seed for shuffling reproducibility. If None, uses random state. Default is None.
        output (str|None): Base path for saving subsets. Each file appended with .0, .1, .2, etc. Default is None.
        ensure_ascii (bool): If True, escapes non-ASCII characters in JSON output. Default is False.
        indent (int|None): JSON indentation level. If None, compact output. Default is 4.

    Returns:
        list[dict]: List of COCO dicts, one per subset, in order matching ratios.

    Raises:
        ValueError: If ratios don't sum to approximately 1.0.

    Examples:
        >>> train, valid, test = split('data.json', ratios=[0.7, 0.2, 0.1], output='split')
        >>> subsets = split(coco_dict, shuffle=True, random_seed=42)
    """
    cocodata = _load_coco(input, image_root=image_root)
    
    if abs(1.0 - sum(ratios)) > 1e-10:
        raise ValueError('The sum of `ratios` should be 1.')
    ratios_cumsum = [0]
    for r in ratios:
        ratios_cumsum.append(r + ratios_cumsum[-1])
    ratios_cumsum[-1] = 1.0

    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(cocodata['images'])

    image_subsets = []
    for i in range(len(ratios)):
        image_subsets.append([])
        n_samples = len(cocodata['images'])
        n_splits = [int(n_samples * r) for r in ratios_cumsum]
        image_subsets[i] = cocodata['images'][n_splits[i]:n_splits[i + 1]]
    
    data_subsets = []
    for i in range(len(image_subsets)):
        data_subset = {
            'images': image_subsets[i],
            'annotations': [ann for ann in cocodata['annotations'] if ann['image_id'] in [im['id'] for im in image_subsets[i]]],
            'categories': cocodata['categories']
        }
        data_subsets.append(data_subset)
    
    if output is not None:
        for i in range(len(data_subsets)):
            cvtk_utils.save_json(data_subsets[i], f'{output}.{i}', indent=indent, ensure_ascii=ensure_ascii)

    return data_subsets


def reindex(
    input: str|dict,
    image_root: str|None=None,
    image_id=True,
    category_id=True,
    output: str|None=None,
    ensure_ascii=False,
    indent=4
) -> dict:
    """Re-index image and category IDs sequentially.

    Renumbers image and/or category IDs to be sequential (1, 2, 3, ...) and updates all
    annotation references accordingly. Useful after removing items or merging datasets.

    Args:
        input (str|dict): COCO annotation data as file path or dict object.
        image_root (str|None): Base directory for image paths. If None, uses paths from annotations. Default is None.
        image_id (bool): If True, re-index image IDs sequentially. Default is True.
        category_id (bool): If True, re-index category IDs sequentially. Default is True.
        output (str|None): File path to save re-indexed data. If None, only returns without saving. Default is None.
        ensure_ascii (bool): If True, escapes non-ASCII characters in JSON output. Default is False.
        indent (int|None): JSON indentation level. If None, compact output. Default is 4.

    Returns:
        dict: Re-indexed COCO annotation data with updated ID references.

    Examples:
        >>> reindexed = reindex('sparse_ids.json', output='dense_ids.json')
        >>> reindexed = reindex(coco_dict, image_id=True, category_id=False)
    """
    cocodata = _load_coco(input, image_root=image_root)
    
    if image_id:
        image_idmap = {}
        for i, image in enumerate(cocodata['images']):
            image_idmap[image['id']] = i + 1
            image['id'] = i + 1
        for ann in cocodata['annotations']:
            ann['image_id'] = image_idmap[ann['image_id']]

    if category_id:
        category_idmap = {}
        for i, category in enumerate(cocodata['categories']):
            category_idmap[category['id']] = i + 1
            category['id'] = i + 1
        for ann in cocodata['annotations']:
            ann['category_id'] = category_idmap[ann['category_id']]

    if output is not None:
        cvtk_utils.save_json(cocodata, output, indent=indent, ensure_ascii=ensure_ascii)
    return cocodata


def remove(
    input: str|dict,
    image_root: str|None=None,
    images: list|None=None,
    categories: list|None=None,
    annotations: list|None=None,
    output: str|None=None,
    ensure_ascii=False,
    indent=4
) -> dict:
    """Remove specific images, categories, or annotations from COCO data.

    Deletes specified items and all related annotations. Related annotations are also removed
    when their parent image or category is deleted. Original IDs are preserved (not re-indexed).

    Args:
        input (str|dict): COCO annotation data as file path or dict object.
        image_root (str|None): Base directory for image paths. If None, uses paths from annotations. Default is None.
        images (list|None): Images to remove. Items can be image IDs (int) or filenames (str). Default is None.
        categories (list|None): Categories to remove. Items can be category IDs (int) or names (str). Default is None.
        annotations (list|None): Annotations to remove by annotation ID (int). Default is None.
        output (str|None): File path to save filtered data. If None, only returns without saving. Default is None.
        ensure_ascii (bool): If True, escapes non-ASCII characters in JSON output. Default is False.
        indent (int|None): JSON indentation level. If None, compact output. Default is 4.

    Returns:
        dict: COCO data with specified items removed.

    Examples:
        >>> remove('data.json', images=[1, 5, 10], output='filtered.json')
        >>> remove(coco_dict, categories=['background'], annotations=[1, 2, 3])
    """
    if isinstance(images, str) or isinstance(images, int):
        images = [images]
    if isinstance(categories, str) or isinstance(categories, int):
        categories = [categories]
    if isinstance(annotations, str) or isinstance(annotations, int):
        annotations = [annotations]
    
    cocodata = _load_coco(input, image_root=image_root)

    rm_images = []
    cocodata_images = copy.deepcopy(cocodata['images'])
    if (images is not None) and (len(images) > 0):
        cocodata_images = []
        for im in cocodata['images']:
            if (im['id'] in images) or (im['file_name'] in images):
                rm_images.append(im['id'])
            else:
                cocodata_images.append(im)
    
    rm_cates = []
    cocodata_cates = copy.deepcopy(cocodata['categories'])
    if (categories is not None) and (len(categories) > 0):
        cocodata_cates = []
        for cate in cocodata['categories']:
            if (cate['id'] in categories) or (cate['name'] in categories):
                rm_cates.append(cate['id'])
            else:
                cocodata_cates.append(cate)

    cocodata_anns = []
    for ann in cocodata['annotations']:
        if (annotations is not None) and (len(annotations) > 0):
            if ann['id'] in annotations:
                continue
        if ann['image_id'] in rm_images:
            continue
        if ann['category_id'] in rm_cates:
            continue
        cocodata_anns.append(ann)
    
    cocodata['images'] = cocodata_images
    cocodata['categories'] = cocodata_cates
    cocodata['annotations'] = cocodata_anns

    if output is not None:
        cvtk_utils.save_json(cocodata, output, indent=indent, ensure_ascii=ensure_ascii)  
    return cocodata


def stats(
    input: str|dict,
    image_root: str|None=None,
    output: str|None=None,
    ensure_ascii: bool=False,
    indent: int|None=4
) -> dict:
    """Calculate dataset statistics from COCO annotations.

    Computes summary statistics including total images, categories, and annotation counts per category.

    Args:
        input (str|dict): COCO annotation data as file path or dict object.
        image_root (str|None): Base directory for image paths. If None, uses paths from annotations. Default is None.
        output (str|None): File path to save statistics. If None, only returns without saving. Default is None.
        ensure_ascii (bool): If True, escapes non-ASCII characters in JSON output. Default is False.
        indent (int|None): JSON indentation level. If None, compact output. Default is 4.

    Returns:
        dict: Statistics containing 'n_images', 'n_categories', and 'n_annotations' (per category).

    Examples:
        >>> stats_data = stats('data.json')
        >>> print(f"Total images: {stats_data['n_images']}")
        >>> print(f"Annotations per class: {stats_data['n_annotations']}")
    """
    cocodata = _load_coco(input, image_root=image_root)

    n_anns = {}
    for cate in cocodata['categories']:
        n_anns[str(cate['id'])] = 0
    for ann in cocodata['annotations']:
        if str(ann['category_id']) not in n_anns:
            raise ValueError(f'Annotation category_id {ann["category_id"]} not found in categories.')
        n_anns[str(ann['category_id'])] += 1

    stats = {
        'n_images': len(cocodata['images']),
        'n_categories': len(cocodata['categories']),
        'n_annotations': {cate['name']: n_anns[str(cate['id'])] for cate in cocodata['categories']}
    }

    if output is not None:
        cvtk_utils.save_json(stats, output, indent=indent, ensure_ascii=ensure_ascii)
    return stats


def __check_list_in_dict(d, k, allow_empty=False):
    if not isinstance(d, dict):
        raise ValueError(f'Expected a dictionary, but got {type(d)}.')
    if k not in d:
        raise ValueError(f'Key "{k}" not found in the dictionary.')
    if d[k] is None:
        raise ValueError(f'The value for key "{k}" is None.')
    if not isinstance(d[k], list):
        raise ValueError(f'Expected a list for key "{k}", but got {type(d[k])}.')
    if len(d[k]) == 0 and not allow_empty:
        raise ValueError(f'The list for key "{k}" is empty.')


def __empty_prediction_stats(metrics_labels, coco_gt):
    stats_ = {label: 0.0 for label in metrics_labels}
    class_stats = {}
    for category in coco_gt.dataset['categories']:
        class_stats[category['name']] = {label: 0.0 for label in metrics_labels}
    return {
        'stats': stats_,
        'class_stats': class_stats
    }


def calc_stats(
    gt: str|dict,
    pred: str|dict,
    image_root: str|None=None,
    image_by: Literal['id', 'file_name']='id',
    category_by: Literal['id', 'name']='id',
    iouType: Literal['bbox', 'segm']='bbox',
    metrics_labels=None
) -> dict:
    """Calculate object detection and segmentation metrics using COCO evaluation.

    Computes standard COCO metrics (AP, AR) for object detection and instance segmentation tasks.
    Supports flexible ID mapping between ground truth and predictions via image filenames or category names.
    Uses pycocotools COCOeval for metric computation.
    
    Args:
        gt (str|dict): Ground truth COCO annotations as file path or dict object.
        pred (str|dict): Predicted COCO annotations as file path or dict object.
        image_root (str|None): Base directory for image paths. If None, uses paths from annotations. Default is None.
        image_by (Literal['id', 'file_name']): Attribute for mapping images between gt and pred:
            - 'id': Match by image ID (must be identical)
            - 'file_name': Match by filename (allows different IDs). Default is 'id'.
        category_by (Literal['id', 'name']): Attribute for mapping categories:
            - 'id': Match by category ID (must be identical)
            - 'name': Match by category name (allows different IDs). Default is 'id'.
        iouType (Literal['bbox', 'segm']): Evaluation type: 'bbox' for object detection, 'segm' for segmentation. Default is 'bbox'.
        metrics_labels (list|tuple|None): Specific metric labels to compute (e.g., ['AP@[0.50:0.95|all|100]']).
            If None, computes all 12 standard COCO metrics. Default is None.

    Returns:
        dict: Metrics with structure {'stats': {...}, 'class_stats': {...}}:
            - 'stats': Overall metrics for each requested label
            - 'class_stats': Per-category metrics with class names as keys

    Raises:
        ValueError: If iouType not in ['bbox', 'segm'], metrics_labels empty, or metric parsing fails.
        TypeError: If metrics_labels not list or tuple.

    Examples:
        >>> results = calc_stats('gt.json', 'pred.json', iouType='bbox')
        >>> print(f"AP: {results['stats']['AP@[0.50:0.95|all|100]']}")
        >>> results = calc_stats(gt_dict, pred_dict, category_by='name', image_by='file_name')
    """
    if iouType not in ['bbox', 'segm']:
        raise ValueError(f'Invalid iouType: {iouType}. Supported types are "bbox" and "segm".')    
    if metrics_labels is None:
        metrics_labels = list(_METRICS_LABELS)
    elif isinstance(metrics_labels, (list, tuple)):
        if len(metrics_labels) == 0:
            raise ValueError('`metrics_labels` should not be empty.')
        metrics_labels = list(metrics_labels)
    else:
        raise TypeError('`metrics_labels` should be list or tuple.')
    metric_tags = __get_metric_tags(metrics_labels)

    # groundtruth
    coco_gt = pycocotools.coco.COCO()
    coco_gt.dataset = _load_coco(gt, image_root=image_root)
    coco_gt.createIndex()

    # prediction
    coco_pred = _load_coco(pred, image_root=image_root)
    __check_list_in_dict(coco_pred, 'annotations', allow_empty=True)
    __check_list_in_dict(coco_pred, 'images')
    __check_list_in_dict(coco_pred, 'categories')

    if len(coco_pred['annotations']) == 0:
        return __empty_prediction_stats(metrics_labels, coco_gt)

    # replace image ID if gt and pred coco have different image ID
    if image_by == 'id':
        # do nothing, since the default mapping is by image ID
        pass
    elif image_by == 'file_name':
        gt_f2id = {str(_['file_name']): _['id'] for _ in coco_gt.dataset['images']}
        pred_id2f = {str(_['id']): _['file_name'] for _ in coco_pred['images']}        
        # replace image ID in annotations
        for i in range(len(coco_pred['annotations'])):
            pred_image_id = str(coco_pred['annotations'][i]['image_id'])
            if pred_image_id not in pred_id2f:
                raise ValueError(f'The annotation {i} is associated with prediction image_id={pred_image_id}, but this image_id is not found in prediction images table.')
            if pred_id2f[pred_image_id] not in gt_f2id:
                raise ValueError(f'The annotation {i} is associated with prediction image_id={pred_image_id}, but the corresponding prediction image is not found in ground truth annotations.')
            coco_pred['annotations'][i]['image_id'] = gt_f2id[pred_id2f[pred_image_id]]
    else:
        raise ValueError('Unsupported mapping type.')

    # replace category ID
    if category_by == 'id':
        pass
    elif category_by == 'name':
        gt_cate2id = {_['name']: _['id'] for _ in coco_gt.dataset['categories']}
        pred_id2cate = {str(_['id']): _['name'] for _ in coco_pred['categories']}
        # replace category ID in annotations
        for i in range(len(coco_pred['annotations'])):
            pred_category_id = str(coco_pred['annotations'][i]['category_id'])
            if pred_category_id not in pred_id2cate:
                raise ValueError(f'Prediction category_id={pred_category_id} is not found in prediction categories table.')
            if pred_id2cate[pred_category_id] not in gt_cate2id:
                raise ValueError(f'Prediction category name "{pred_id2cate[pred_category_id]}" is not found in ground truth categories.')
            coco_pred['annotations'][i]['category_id'] = gt_cate2id[pred_id2cate[pred_category_id]]
    else:
        raise ValueError(f'Unsupported category mapping type: {category_by}')

    if iouType == 'segm':
        for i, ann in enumerate(coco_pred['annotations']):
            # Segmentation can be either RLE (dict) or polygon (list)
            if 'segmentation' not in ann:
                raise ValueError(f'Annotation {i} missing "segmentation" key.')
            if ann['segmentation'] is None:
                raise ValueError(f'Annotation {i} has None segmentation.')
            if not isinstance(ann['segmentation'], (list, dict)):
                raise ValueError(f'Annotation {i} segmentation must be list (polygon) or dict (RLE), but got {type(ann["segmentation"])}.')
            if isinstance(ann['segmentation'], list) and len(ann['segmentation']) == 0:
                raise ValueError(f'Annotation {i} has empty segmentation list.')
            # avoid that pycocotools priorily loads bbox (may use bbox as segmentation coordinates)
            ann.pop('bbox', None)
    
    coco_pred_anns = coco_gt.loadRes(coco_pred['annotations'])
    coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_pred_anns, iouType=iouType)
    coco_eval.params.maxDets = metric_tags['maxDets']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    parsed_metrics = {}
    for label in metrics_labels:
        parsed_metrics[label] = __parse_metrics_label(label, coco_eval)

    # overall stats
    stats_ = {}
    for label in metrics_labels:
        stats_[label] = __compute_metric(coco_eval, parsed_metrics[label], class_idx=None)

    # class stats
    class_stats = {}
    for cat_idx, cat_id in enumerate(list(coco_eval.params.catIds)):
        category_name = coco_gt.loadCats(int(cat_id))[0]['name']
        class_stats[category_name] = {}
        for label in metrics_labels:
            class_stats[category_name][label] = __compute_metric(coco_eval, parsed_metrics[label], class_idx=cat_idx)

    return {
        'stats': stats_,
        'class_stats': class_stats
    }
    


def __get_metric_tags(metric_labels):
    # expected: ['AP@[0.50:0.95|all|1000] or AR@[0.75|medium|300]', ...]
    metric_types = set()
    iou_thrs = set()
    objsizes = set()
    maxdets = set()
    
    for metric_label in metric_labels:
        if '@[' not in metric_label or not metric_label.endswith(']'):
            raise ValueError(f'Invalid metric label format: {metric_label}')
        metric_type, spec = metric_label.split('@[', 1)
        metric_types.add(metric_type.strip())
    
        metric_parts = spec[:-1].split('|')
        if len(metric_parts) != 3:
            raise ValueError(f'Invalid metric label format: {metric_label}')
        
        iou_thrs.add(metric_parts[0].strip())
        objsizes.add(metric_parts[1].strip())
        maxdets.add(int(metric_parts[2].strip()))
        
    return {
        'metricTypes': sorted(list(metric_types)),
        'iouThrs': sorted(list(iou_thrs)),
        'objSizes': sorted(list(objsizes)),
        'maxDets': sorted(list(maxdets))
    }
 

def __parse_metrics_label(metric_label, coco_eval):
    metric_type, spec = metric_label.split('@[', 1)
    metric_type = metric_type.strip()

    parts = spec[:-1].split('|')
    iou_spec = parts[0].strip()
    area_label = parts[1].strip()
    max_det = int(parts[2].strip())
    
    iou_thrs = np.asarray(coco_eval.params.iouThrs, dtype=float)
    area_labels = list(coco_eval.params.areaRngLbl)
    max_dets = list(coco_eval.params.maxDets)
    
    if metric_type not in ['AP', 'AR']:
        raise ValueError(f'Unsupported metric type: {metric_type}. Supported types are "AP" and "AR".')

    if area_label not in area_labels:
        raise ValueError(f'Area label "{area_label}" not found in evaluator area labels: {area_labels}')
    area_idx = area_labels.index(area_label)

    if max_det not in max_dets:
        raise ValueError(f'Max detection {max_det} not found in evaluator max detections: {max_dets}')
    max_det_idx = max_dets.index(max_det)
    
    if ':' in iou_spec:
        range_parts = iou_spec.split(':', 1)
        try:
            iou_min = float(range_parts[0])
            iou_max = float(range_parts[1])
        except ValueError as e:
            raise ValueError(f'Invalid IoU range in metric label: {metric_label}') from e

        iou_idx = [i for i, thr in enumerate(iou_thrs)
                   if (thr >= iou_min - 1e-9) and (thr <= iou_max + 1e-9)]
        if len(iou_idx) == 0:
            raise ValueError(f'No IoU thresholds found in range [{iou_min}, {iou_max}] for label: {metric_label}')
    else:
        try:
            iou_val = float(iou_spec)
        except ValueError as e:
            raise ValueError(f'Invalid IoU value in metric label: {metric_label}') from e

        iou_idx = [i for i, thr in enumerate(iou_thrs) if np.isclose(thr, iou_val, atol=1e-9)]
        if len(iou_idx) == 0:
            raise ValueError(f'IoU threshold {iou_val} not found in evaluator thresholds: {iou_thrs.tolist()}')

    return {
            'metric_type': metric_type,
            'iou_idx': iou_idx,
            'area_idx': area_idx,
            'max_det_idx': max_det_idx
        }
    

def __compute_metric(coco_eval, parsed_metric, class_idx=None):
    metric_type = parsed_metric['metric_type']
    iou_idx = parsed_metric['iou_idx']
    area_idx = parsed_metric['area_idx']
    max_det_idx = parsed_metric['max_det_idx']

    if metric_type == 'AP':
        precision = coco_eval.eval['precision']
        if class_idx is None:
            vals = precision[iou_idx, :, :, area_idx, max_det_idx]
        else:
            vals = precision[iou_idx, :, class_idx, area_idx, max_det_idx]
    else:
        recall = coco_eval.eval['recall']
        if class_idx is None:
            vals = recall[iou_idx, :, area_idx, max_det_idx]
        else:
            vals = recall[iou_idx, class_idx, area_idx, max_det_idx]

    vals = np.asarray(vals)
    vals = vals[vals > -1]
    if vals.size == 0:
        return float('nan')
    return float(vals.mean())
