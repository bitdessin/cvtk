import random
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
