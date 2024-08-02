import os
import shutil
import importlib
import random
import copy
import re
from .torchutils import __generate_source as generate_source_cls
from .mmdetutils import __generate_source as generate_source_det
from ._subutils import __estimate_task_from_source, __generate_app_html_tmpl


def split_dataset(data: str|list[str, str]|tuple[str, str],
                  output: str|None=None,
                  ratios: list[float]|tuple[float]=[0.8, 0.1, 0.1],
                  shuffle: bool=True,
                  stratify: bool=True,
                  random_seed: int|None=None) -> list[list]:
    """Split a dataset into multiple subsets with the given ratios

    Split a dataset into multiple subsets with the given ratios.
    
    Args:
        data: The dataset to split. The input can be a list of data (e.g., images)
            or a path to a text file. If list is given, each element of the list is treated as a sample.
        output: The output file name will be appended with the index of the split subset.
        ratios: The ratios to split the dataset. The sum of the ratios should be 1.
        shuffle (bool): Shuffle the dataset before splitting.
        stratify (bool): Split the dataset with a balanced class distribution if `label` is given.
        random_seed (int|none): Random seed for shuffling the dataset.

    Returns:
        A list of the split datasets. The length of the list is the same as the length of `ratios`.

    Examples:
        >>> from cvtk.ml import split_dataset
        >>> 
        >>> subsets = split_dataset('data.txt', ratios=[0.8, 0.1, 0.1])
        >>> len(subsets)
        3
    """
    if abs(1.0 - sum(ratios)) > 1e-10:
        raise ValueError('The sum of `ratios` should be 1.')

    data_ = []
    label_ = []
    if isinstance(data, str):
        with open(data, 'r') as infh:
            for line in infh:
                line = line.strip()
                m = line.split('\t', 2)
                data_.append(line)
                if len(m) > 1:
                    label_.append(m[1])
    elif isinstance(data, (list, tuple)):
        for d in data:
            data_.append(d)
            if len(d) > 1:
                label_.append(d[1])
    else:
        raise ValueError('The input data should be a list or a path to a text file.')
    data = data_
    label = label_

    ratios_cumsum = [0]
    for r in ratios:
        ratios_cumsum.append(r + ratios_cumsum[-1])
    ratios_cumsum[-1] = 1.0

    # shuflle data
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        idx = list(range(len(data)))
        random.shuffle(idx)
        data = [data[i] for i in idx]
        if len(label) > 0:
            label = [label[i] for i in idx]
    
    # group data by label
    datadict = {}
    if stratify and len(label) > 0:
        for i, label in enumerate(label):
            if label not in datadict:
                datadict[label] = []
            datadict[label].append(data[i])
    
    # split data
    data_subsets = []
    label_subsets = []
    for i in range(len(ratios)):
        data_subsets.append([])
        label_subsets.append([])
        if len(datadict) > 0:
            for cl in datadict:
                n_samples = len(datadict[cl])
                n_splits = [int(n_samples * r) for r in ratios_cumsum]
                data_subsets[i] += datadict[cl][n_splits[i]:n_splits[i + 1]]
                label_subsets[i] += [cl] * (n_splits[i + 1] - n_splits[i])
        else:
            n_samples = len(data)
            n_splits = [int(n_samples * r) for r in ratios_cumsum]
            data_subsets[i] = data[n_splits[i]:n_splits[i + 1]]

    if output is not None:
        for i in range(len(data_subsets)):
            with open(f'{output}.{i}', 'w') as fh:
                for data_record in data_subsets[i]:
                    if isinstance(data_record, (list, tuple)):
                        data_record = '\t'.join(data_record)
                    fh.write(data_record + '\n')
    
    return data_subsets



def generate_source(project: str, task: str='cls', vanilla=False) -> None:
    """Generate source code for training and inference of a classification model using PyTorch

    This function generates a Python script for training and inference of a model
    using PyTorch (for classification task) or MMDetection (for object detection and instance segmentation tasks).
    Two types of scripts can be generated based on the `vanilla` argument:
    one with importation of cvtk and the other without importation of cvtk.
    The script with importation of cvtk keeps the code simple and easy to understand,
    since most complex functions are implemented in cvtk.
    It designed for users who are beginning to learn deep learning for image tasks with PyTorch or MMDetection.
    On the other hand, the script without cvtk import is longer and more exmplex,
    but it can be more flexibly customized and further developed, 
    since all functions is implemented directly in torch and torchvision.

    Args:
        project (str): A file path to save the script.
        task (str): The task type of project. Three types of tasks can be specified ('cls', 'det', 'segm'). The default is 'cls'.
        vanilla (bool): Generate a script without importation of cvtk. The default is False.
    """
    
    if task.lower() in ['cls', 'classification']:
        generate_source_cls(project, vanilla)
    elif task.lower() in ['det', 'detection', 'seg', 'segm', 'segmentation']:
        generate_source_det(project, task, vanilla)
    else:
        raise ValueError('The current version only support classification (`cls`), detection (`det`), and segmentation (`segm`) tasks.')


def generate_app(project: str, source: str, label: str, model: str, weights: str, vanilla=False) -> None:
    """Generate a FastAPI application for inference of a classification or detection model
    
    This function generates a FastAPI application for inference of a classification or detection model.

    Args:
        project (str): A file path to save the FastAPI application.
        source (str): The source code of the model.
        label (str): The label file of the dataset.
        model (str): The configuration file of the model.
        weights (str): The weights file of the model.
        module (str): Script with importation of cvtk ('cvtk') or not ('fastapi').

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

    source_task_type, source_is_vanilla = __estimate_task_from_source(source)

    # FastAPI script
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/_fastapi.py'), source_task_type)
    if vanilla:
        if source_is_vanilla:
            for i in range(len(tmpl)):
                if tmpl[i][:9] == 'from cvtk':
                    if source_task_type == 'cls':
                        tmpl[i] = f'from {coremodule} import CLSCORE as MODULECORE'
                    elif source_task_type == 'det':
                        tmpl[i] = f'from {coremodule} import MMDETCORE as MODULECORE'
                    else:
                        raise ValueError('Unsupport Type.')
        else:
            print('The CLSCORE or MMDETCORE class definition is not found in the source code. The script will be generated with importation of cvtk.')
    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__DATALABEL__', data_label)
    tmpl = tmpl.replace('__MODELCFG__', model_cfg)
    tmpl = tmpl.replace('__MODELWEIGHT__', model_weights)
    with open(os.path.join(project, 'main.py'), 'w') as fh:
        fh.write(tmpl)

    # HTML template
    if not os.path.exists(os.path.join(project, 'templates')):
        os.makedirs(os.path.join(project, 'templates'))
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/html/fastapi_.html'), source_task_type)
    with open(os.path.join(project, 'templates', 'index.html'), 'w') as fh:
        fh.write(''.join(tmpl))
    