import os
import shutil
import importlib
import random
import re
from .torch import __generate_source as generate_source_cls
from .mmdet import __generate_source as generate_source_det


def split_dataset(data, label=None, ratios=[0.8, 0.1, 0.1], balanced=True, shuffle=True, random_seed=None):
    """Split a dataset into train, validation, and test sets

    Split a dataset into several subsets with the given ratios.
    
    
    Args:
        data (str|list): The dataset to split. The input can be a list of data (e.g., images)
            or a path to a text file.
        labels (list): The labels corresponding to the `data`.
        ratios (list): The ratios to split the dataset. The sum of the ratios should be 1.
        balanced (bool): Split the dataset with a balanced class distribution if `label` is given.
        shuffle (bool): Shuffle the dataset before splitting.
        random_seed (int): Random seed for shuffling the dataset.

    Returns:
        A list of the split datasets. The length of the list is the same as the length of `ratios`.

    Examples:
        >>> from cvtk.ml import split_dataset
        >>> 
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        >>> train_data, val_data, test_data, train_labels, val_labels, test_labels = split_dataset(data, labels)
    """
    data_from_file = False
    if isinstance(data, str):
        data_ = []
        label_ = []
        with open(data, 'r') as infh:
            for line in infh:
                line = line.strip()
                m = line.split('\t', 2)
                data_.append(line)
                if len(m) > 1:
                    label_.append(m[1])
        data = data_
        if len(label_) > 0:
            label = label_
        data_from_file = True

    if label is not None and len(data) != len(label):
        raise ValueError('The length of `data` and `labels` should be the same.')
    if abs(1.0 - sum(ratios)) > 1e-10:
        raise ValueError('The sum of `ratios` should be 1.')
    ratios_cumsum = [0]
    for r in ratios:
        ratios_cumsum.append(r + ratios_cumsum[-1])
    ratios_cumsum[-1] = 1
    
    dclasses = {}
    if label is not None:
        for i, label in enumerate(label):
            if label not in dclasses:
                dclasses[label] = []
            dclasses[label].append(data[i])
    else:
        dclasses['__ALLCLASSES__'] = data
    
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        for cl in dclasses:
            random.shuffle(dclasses[cl])
    
    data_subsets = []
    label_subsets = []
    for i in range(len(ratios)):
        data_subsets.append([])
        label_subsets.append([])
        if balanced:
            for cl in dclasses:
                n_samples = len(dclasses[cl])
                n_splits = [int(n_samples * r) for r in ratios_cumsum]
                data_subsets[i] += dclasses[cl][n_splits[i]:n_splits[i + 1]]
                label_subsets[i] += [cl] * (n_splits[i + 1] - n_splits[i])
        else:
            n_samples = len(data)
            n_splits = [int(n_samples * r) for r in ratios_cumsum]
            data_subsets[i] = data[n_splits[i]:n_splits[i + 1]]
            if label is not None:
                label_subsets[i] = label[n_splits[i]:n_splits[i + 1]]
    
    if data_from_file or (label is None):
        return data_subsets
    else:
        return data_subsets, label_subsets





def generate_source(project, task='classification', module='cvtk'):
    """Generate source code for training and inference of a classification model using PyTorch

    This function generates a Python script for training and inference of a classification model using PyTorch.
    Two types of scripts can be generated based on the `module` argument:
    one with importation of cvtk and the other without importation of cvtk.
    The script with importation of cvtk keeps the code simple and easy to understand,
    since most complex functions are implemented in cvtk.
    It designed for users who are beginning to learn object classification with PyTorch.
    On the other hand, the script without cvtk import is longer and more exmplex,
    but it can be more flexibly customized and further developed, 
    since all functions is implemented directly in torch and torchvision.

    Args:
        project (str): A file path to save the script.
        task (str): The task type of project. Only 'classification' is supported in the current version.
        module (str): Script with importation of cvtk ('cvtk') or not ('torch').
    """
    if task.lower() in ['cls', 'classification']:
        generate_source_cls(project, module)
    elif task.lower() in ['det', 'detection', 'seg', 'segm', 'segmentation', 'mmdet', 'mmdetection']:
        generate_source_det(project, task, module)
    else:
        raise ValueError('The current version only support classification (`cls`), detection (`det`), and segmentation (`seg`) tasks.')


def generate_app(project, source, label, model, weights, module='cvtk'):
    """Generate a FastAPI application for inference of a classification or detection model
    
    This function generates a FastAPI application for inference of a classification or detection model.

    Args:
        project (str): A file path to save the FastAPI application.
        source (str): The source code of the model.
        label (str): The label file of the dataset.
        model (str): The configuration file of the model.
        weights (str): The weights file of the model.
        module (str): The module name of the model. The default is 'cvtk'.

    Examples:
        >>> from cvtk.ml import generate_app
        >>> generate_app('./project', 'model.py', 'label.txt', 'model.cfg', 'model.pth')
    
    """

    if not os.path.exists(project):
        os.makedirs(project)

    coremodule = os.path.splitext(os.path.basename(source))[0]
    data_label = os.path.basename(label)
    model_cfg = os.path.basename(model)
    model_weights = os.path.basename(weights)

    shutil.copy2(source, os.path.join(project, coremodule + '.py'))
    shutil.copy2(label, os.path.join(project, data_label))
    if os.path.exists(model):
        shutil.copy2(model, os.path.join(project, model_cfg))
    shutil.copy2(weights, os.path.join(project, model_weights))

    task = __estimate_task_from_source(source)

    # FastAPI script
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/fastapi_.py'), task)
    if module != 'cvtk':
        for i in range(len(tmpl)):
            if tmpl[i][:9] == 'from cvtk':
                if task == 'cls':
                    tmpl[i] = f'from {coremodule} import CLSCORE as MODULECORE'
                elif task == 'det':
                    tmpl[i] = f'from {coremodule} import MMDETCORE as MODULECORE'
                else:
                    raise ValueError('Unsupport Type.')
    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__DATALABEL__', data_label)
    tmpl = tmpl.replace('__MODELCFG__', model_cfg)
    tmpl = tmpl.replace('__MODELWEIGHT__', model_weights)
    with open(os.path.join(project, 'main.py'), 'w') as fh:
        fh.write(tmpl)

    # HTML template
    if not os.path.exists(os.path.join(project, 'templates')):
        os.makedirs(os.path.join(project, 'templates'))
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/html/fastapi_.html'), task)
    with open(os.path.join(project, 'templates', 'index.html'), 'w') as fh:
        fh.write(''.join(tmpl))
    

    
def __generate_app_html_tmpl(tmpl_fpath, task):
    tmpl = []

    write_code = True
    with open(tmpl_fpath, 'r') as infh:
        for codeline in infh:
            if '#%CVTK%#' in codeline:
                if ' IF' in codeline:
                    m = re.search(r'TASK=([^\s\}]+)', codeline)
                    task_code = m.group(1) if m else None
                    if task_code is None:
                        raise ValueError('Unable to get task code.')
                    if task in task_code:
                        write_code = True
                    else:
                        write_code = False
                elif ' ENDIF' in codeline:
                    write_code = True
                continue

            if write_code:
                tmpl.append(codeline)

    return tmpl


def __estimate_task_from_source(source):
    task_ = {
        'CLSCORE': {'classdef': 0, 'import': 0, 'call': 0},
        'MMDETCORE': {'classdef': 0, 'import': 0, 'call': 0}
    }
    with open(source, 'r') as infh:
        for codeline in infh:
            if 'class CLSCORE' in codeline:
                task_['CLSCORE']['classdef'] += 1
            elif 'import cvtk.ml.torch' in codeline:
                task_['CLSCORE']['import'] += 1
            elif 'from cvtk.ml.torch import' in codeline and 'CLSCORE' in codeline:
                task_['CLSCORE']['import'] += 1
            elif 'CLSCORE(' in codeline:
                task_['CLSCORE']['call'] += 1
            elif 'class MMDETCORE' in codeline:
                task_['MMDETCORE']['classdef'] += 1
            elif 'import cvtk.ml.mmdet' in codeline:
                task_['MMDETCORE']['import'] += 1
            elif 'from cvtk.ml.mmdet import' in codeline and 'MMDETCORE' in codeline:
                task_['MMDETCORE']['import'] += 1
            elif 'MMDETCORE(' in codeline:
                task_['MMDETCORE']['call'] += 1              
    
    is_task_cls = ((task_['CLSCORE']['classdef'] > 0) or (task_['CLSCORE']['import'] > 0)) and (task_['CLSCORE']['call'] > 0)
    is_task_det = ((task_['MMDETCORE']['classdef'] > 0) or (task_['MMDETCORE']['import'] > 0)) and (task_['MMDETCORE']['call'] > 0)

    if is_task_cls and not is_task_det:
        task = 'cls'
    elif not is_task_cls and is_task_det:
        task = 'det'
    else:
        raise ValueError('The task type cannot be determined from the source code. Make sure your source code contains CLSCORE or MMDETCORE class definition or importation, and call.')

    return task
