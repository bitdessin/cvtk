import os
import json
import random
from .base import JsonComplexEncoder


def merge(inputs: str|list[str], output: str|None=None, ensure_ascii: bool=False, indet: int|None=4) -> dict:
    """Merge multiple COCO annotation files into one file.

    The function will merge the images, annotations, and categories
    from multiple COCO annotation files into one file.
    The IDs of the images, annotations, and categories will be re-indexed.


    Args:
        inputs (list[str]): List of file paths to COCO annotation files to be merged.
        output (str, None): The merged COCO annotation data will be saved to the file if the file path is given.
        ensure_ascii (bool): If True, the output is guaranteed to have all incoming non-ASCII characters escaped.
        indet (int, None): If a non-negative integer is provided,
            the output JSON data will be formatted with the given indentation.

    Returns:
        dict: Merged COCO annotation data.
    
    Examples:
        >>> merged_coco = merge(['annotations1.json', 'annotations2.json', 'annotations3.json'],
                                'merged_annotations.json')
    """

    merged_coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    image_id = 1
    category_id = 1
    annotation_id = 1
    
    for input_file in inputs:
        image_idmap = {}
        category_idmap = {}

        with open(input_file, 'r') as f:
            data = json.load(f)
            
            for category in data['categories']:
                if category['name'] not in [c['name'] for c in merged_coco['categories']]:
                    category_idmap[category['id']] = category_id
                    category['id'] = category_id
                    merged_coco['categories'].append(category.copy())
                    category_id += 1
                else:
                    category_idmap[category['id']] = [c['id'] for c in merged_coco['categories'] if c['name'] == category['name']][0]
            
            for image in data['images']:
                image_idmap[image['id']] = image_id
                image['id'] = image_id
                merged_coco['images'].append(image.copy())
                image_id += 1
            
            for annotation in data['annotations']:
                annotation['id'] = annotation_id
                annotation['image_id'] = image_idmap[annotation['image_id']]
                annotation['category_id'] = category_idmap[annotation['category_id']]
                merged_coco['annotations'].append(annotation)
                annotation_id += 1
    
    if output:
        with open(output, 'w') as f:
            json.dump(merged_coco, f, cls=JsonComplexEncoder, ensure_ascii=ensure_ascii, indent=indet)
    
    return merged_coco




def calc_stats(gt: str|dict, pred: str|dict, image_by: str='id', category_by='id', iouType: str='bbox', metrics_labels=None) -> dict:
    """Calculate prediction performance metrics for object detection and instance segmentation tasks

    The function calculates the prediction performance metrics for object detection and instance segmentation tasks,
    using the COCO evaluation API from pycocotools.
    The ground truth and predicted annotations can be provided as file paths or dict objects of COCO annotations.
    The image IDs between the ground truth and prediction can be different;
    the function provides an option to map them by filepath or filename by specifying the `image_by` parameter.
    In addition, the category IDs between the ground truth and prediction can be different;
    the function provides an option to map them by category name by specifying the `category_by` parameter.
    
    Args:
        gt (str|dict): Annotations of ground truth. It can be a path to a COCO annotation file or a dict object of COCO annotation.
        pred (str|dict): The predicted annotations. It can be a path to a COCO annotation file or a dict object of COCO annotation.
        image_by (str): The attribute to map image ID between ground truth and prediction. Default is 'id'.
        category_by (str): The attribute to map category ID between ground truth and prediction. Default is 'id'.
        iouType (str): The type of IoU calculation. Default is 'bbox', but 'segm' is also available.
        
    Returns:
        dict: A dictionary containing the prediction performance metrics.

    Examples:
        >>> calculate_map('ground_truth.json', 'predictions.json')
    """
    try:
        import pycocotools.coco
        import pycocotools.cocoeval
    except ImportError as e:
        raise ImportError('Unable to import pycocotools module. '
                          'Install pycocotools module to enable calculation of prediction performance.') from e

    
    if metrics_labels is None:
        metrics_labels = ['AP@[0.50:0.95|all|100]',
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
        
    
    def __calc_class_stats(coco_eval, coco_gt):
        metrics = {}

        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        area_ranges = ['all', 'small', 'medium', 'large']
        max_detections = [1, 10, 100, 300, 1000]

        for cat_id in coco_gt.getCatIds():
            category_name = coco_gt.loadCats(cat_id)[0]['name']
            metrics[category_name] = {}
            
            for i, metric_label in enumerate(metrics_labels):
                metric_label_ = metric_label.replace('[', '').replace(']', '')
                if '0.50:0.95' in metric_label_:
                    iou_thr = slice(None)
                else:
                    iou_thr = iou_thresholds.index(float(metric_label_.split('@')[1].split('|')[0]))
                area = area_ranges.index(metric_label_.split('|')[1])
                max_det = max_detections.index(int(metric_label_.split('|')[2]))
                if 'AP@' in metric_label_:
                    v = coco_eval.eval['precision'][iou_thr, :, cat_id - 1, area, max_det].mean() 
                elif 'AR@' in metric_label_:
                    v = coco_eval.eval['recall'][iou_thr, cat_id - 1, area, max_det].mean()
                metrics[category_name][metric_label] = v

        return metrics


    # groundtruth
    if isinstance(gt, str):
        coco_gt = pycocotools.coco.COCO(gt)
    else:
        coco_gt = pycocotools.coco.COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()

    # predcition
    pred_anns = None
    if isinstance(pred, str):
        with open(pred, 'r') as f:
            pred = json.load(f)
    if isinstance(pred, dict): 
        if 'annotations' in pred:
            pred_anns = pred['annotations']
        else:
            pred_anns = pred

    # replace image ID
    image_by = image_by.replace('_', '')
    if image_by == 'id':
        pass
    elif image_by == 'filepath' or image_by == 'filename':
        # ground truth image ID
        im2id_gt = {}
        for cocoimg in coco_gt.dataset['images']:
            if image_by == 'filepath':
                im2id_gt[cocoimg['file_name']] = cocoimg['id']
            else:
                im2id_gt[os.path.basename(cocoimg['file_name'])] = cocoimg['id']
        # prediction image ID
        id2im_pred = {}
        for cocoimg in pred['images']:
            if image_by == 'filepath':
                id2im_pred[str(cocoimg['id'])] = cocoimg['file_name']
            else:
                id2im_pred[str(cocoimg['id'])] = os.path.basename(cocoimg['file_name'])        
        # replace image ID in annotations
        for cocoann in pred_anns:
            cocoann['image_id'] = im2id_gt[id2im_pred[str(cocoann['image_id'])]]
    else:
        raise ValueError('Unsupport mapping type.')

    # replace category ID
    if category_by == 'name':
        cate2id_gt = {}
        for cococate in coco_gt.dataset['categories']:
            cate2id_gt[cococate['name']] = cococate['id']
        id2cate_pred = {}
        for cococate in pred['categories']:
            id2cate_pred[str(cococate['id'])] = cococate['name']
        for cocoann in pred_anns:
            cocoann['category_id'] = cate2id_gt[id2cate_pred[str(cocoann['category_id'])]]

    coco_pred = coco_gt.loadRes(pred_anns)
    coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_pred, iouType)
    coco_eval.params.maxDets = [1, 10, 100, 300, 1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_ = {}
    for l_, s_ in zip(metrics_labels, coco_eval.stats):
        stats_[l_] = s_

    stats_dict = {
        'stats': stats_,
        'class_stats': __calc_class_stats(coco_eval, coco_gt)
    }

    return stats_dict
    



#TODO check
def split(input_file: str, output_dir: str, split_ratios: list[float]=[0.8, 0.1, 0.1], shuffle: bool=True) -> None:
    """Split a COCO annotation file into train, validation, and test sets

    Args:
        input_file: str: Path to the input COCO annotation file.
        output_dir: str: Path to the output directory.
        split_ratios: list: List of split ratios. Default is [0.8, 0.1, 0.1].
        shuffle: bool: Shuffle the dataset before splitting. Default is True.

    Returns:
        None

    Examples:
        >>> split('annotations.json', 'output_dir', [0.8, 0.1, 0.1])
    """

    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if shuffle:
        random.shuffle(data['images'])
    
    split_indices = [0] + [int(len(data['images']) * r) for r in split_ratios]
    split_indices[-1] = len(data['images'])
    
    for i, split_ratio in enumerate(split_ratios):
        split_data = {
            'images': data['images'][split_indices[i]:split_indices[i+1]],
            'annotations': [ann for ann in data['annotations'] if ann['image_id'] in [im['id'] for im in split_data['images']]],
            'categories': data['categories']
        }
        
        with open(output_dir + '/split_{}.json'.format(i), 'w') as f:
            json.dump(split_data, f, cls=JsonComplexEncoder, ensure_ascii=False, indent=4)
    
    return


def reindex(input_file, ouput_file, classes):
    pass
