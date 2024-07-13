import json
from .base import __JsonEncoder
try:
    import pycocotools.coco
    import pycocotools.cocoeval
except ImportError as e:
    raise ImportError('Unable to import pycocotools. '
                      'Install pycocotools package to enable this feature.') from e



def merge(inputs: str | list[str], output: str | None = None, indet=4) -> dict:
    """Merge multiple COCO annotation files into one

    Args:
        inputs: list: List of input COCO annotation files.
        output: str: Output merged annotation file.

    Returns:
        dict: Merged COCO annotation data.
    
    Examples:
        >>> merge(['annotations1.json', 'annotations2.json', 'annotations3.json'], 'merged_annotations.json')
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
            json.dump(merged_coco, f, cls=__JsonEncoder, ensure_ascii=False, indent=indet)
    
    return merged_coco



__metrics_labels = ['AP@[0.50:0.95|all|100]',
                   'AP@[0.50|all|100]',
                   'AP@[0.75|all|100]',
                   'AP@[0.50:0.95|small|100]',
                   'AP@[0.50:0.95|medium|100]',
                   'AP@[0.50:0.95|large|100]',
                   'AR@[0.50:0.95|all|1]',
                   'AR@[0.50:0.95|all|10]',
                   'AR@[0.50:0.95|all|100]',
                   'AR@[0.50:0.95|small|100]',
                   'AR@[0.50:0.95|medium|100]',
                   'AR@[0.50:0.95|large|100]']


def calc_stats(gt: str | dict, pred: str | dict, iouType: str='bbox') -> list:
    """Calculate mean average precision (mAP) using COCO API
    
    Args:
        gt: str | dict: Path to ground truth annotation file or dict object of ground truth annotation.
        pred: str | dict: Path to prediction file or dict object of prediction result.
        iouType: str: Evaluation type. Default is 'bbox'.

    Returns:
        list of float: List of mAP statistics.

    Examples:
        >>> calculate_map('ground_truth.json', 'predictions.json')
    """

    if isinstance(gt, str):
        coco_gt = pycocotools.coco.COCO(gt)
    else:
        coco_gt = pycocotools.coco.COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()

    imfname2id = {}
    for im in coco_gt.dataset['images']:
        imfname2id[im['file_name']] = im['id']

    if isinstance(pred, str):
        with open(pred, 'r') as f:
            pred = json.load(f)
        imid2fname = {}
        for im in pred['images']:
            imid2fname[str(im['id'])] = im['file_name']
        for ann in pred['annotations']:
            ann['image_id'] = imfname2id[imid2fname[str(ann['image_id'])]]
    if isinstance(pred, dict) and 'annotations' in pred:
        pred = pred['annotations']
    
    coco_pred = coco_gt.loadRes(pred)
    coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_pred, iouType)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_ = {}
    for l_, s_ in zip(__metrics_labels, coco_eval.stats):
        stats_[l_] = s_

    stats_dict = {
        'stats': stats_,
        'class_stats': __calc_class_stats(coco_eval, coco_gt)
    }

    return stats_dict
    


def __calc_class_stats(coco_eval, coco_gt):
    metrics = {}

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    area_ranges = ['all', 'small', 'medium', 'large']
    max_detections = [1, 10, 100]

    for cat_id in coco_gt.getCatIds():
        category_name = coco_gt.loadCats(cat_id)[0]['name']
        metrics[category_name] = {}
        
        for i, metric_label in enumerate(__metrics_labels):
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

