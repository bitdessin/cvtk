import os
import random
import gzip
import contextlib
import filetype
import gc
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision
import torchvision.transforms.v2
from cvtk.ml.data import DataLabel


def _resolve_image_path(img_path, image_base):
    if img_path is None or img_path == '':
        return img_path
    if os.path.isabs(img_path):
        return img_path
    return os.path.join(image_base, img_path)


class DataTransform():
    """Image preprocessing pipeline for classification tasks.

    Composes common image preprocessing transforms for classification using torchvision.
    Provides separate pipelines for training (with augmentation) and inference (without augmentation).
    Intended for users who want a quick preprocessing setup. For advanced customization,
    use torchvision.transforms.Compose directly.
          
    Args:
        shape (int|tuple[int,int]): Target image resolution. If int, creates square images.
        is_train (bool): If True, creates training pipeline with augmentation (random crop, flip, rotation).
            If False, creates inference pipeline (resize only). Default is False.

    Attributes:
        pipeline (torchvision.transforms.Compose): The composed transform pipeline.

    Examples:
        >>> from cvtk.ml.torchutils import DataTransform
        >>> transform_train = DataTransform(224, is_train=True)
        >>> print(transform_train.pipeline)
        >>> transform_inference = DataTransform(224)
        >>> print(transform_inference.pipeline)
    """
    def __init__(self, shape: int|tuple[int, int], is_train=False):
        if isinstance(shape, int):
            shape = (shape, shape)
        elif isinstance(shape, list):
            shape = tuple(shape)
        
        if is_train:
            self.pipeline = torchvision.transforms.Compose([
                    torchvision.transforms.v2.ToImage(),
                    torchvision.transforms.v2.Resize(size=(shape[0] + 50, shape[1] + 50), antialias=True),
                    torchvision.transforms.v2.RandomResizedCrop(size=shape, scale=(0.8, 1.0), antialias=True),
                    torchvision.transforms.v2.RandomHorizontalFlip(0.5),
                    torchvision.transforms.v2.RandomAffine(45),
                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        else:
            self.pipeline = torchvision.transforms.Compose([
                    torchvision.transforms.v2.ToImage(),
                    torchvision.transforms.v2.Resize(size=shape, antialias=True),
                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])



def Dataset(datalabel, dataset, transform, stream_data=False, oversample=False, image_root=None):
    """Create a dataset for image classification.
    
    Factory function that creates either a standard or iterable dataset based on arguments.
    Automatically extracts pipeline from DataTransform objects and handles image loading
    from directories, lists, tuples, or tab-separated files.
    
    Args:
        datalabel (DataLabel|str|list|tuple): Class labels. Can be a DataLabel instance,
            file path, or list/tuple of label names.
        dataset (str|list|tuple): Image data source:
            - File path: TSV file, image directory, or single image file
            - List/tuple: Image paths with optional labels as nested lists/tuples
        transform (DataTransform|torchvision.transforms.Compose|None): Image preprocessing pipeline.
        stream_data (bool): If True, returns iterable dataset for memory-efficient streaming.
            If False, returns standard dataset that loads all at once. Default is False.
        oversample (bool): If True, oversample minority classes to balance dataset.
            Only works with labeled data. Default is False.
        image_root (str|None): Base directory for relative image paths in dataset.
            If None, uses directory of dataset file (for TSV) or current directory. Default is None.
    
    Returns:
        Dataset_|DatasetIterable_: A PyTorch dataset object ready for DataLoader.
    
    Examples:
        >>> from cvtk.ml import DataLabel
        >>> from cvtk.ml.torchutils import Dataset, DataTransform
        >>> datalabel = DataLabel(['cat', 'dog'])
        >>> transform = DataTransform(224, is_train=True)
        >>> dataset = Dataset(datalabel, 'train.txt', transform)
        >>> print(len(dataset))
    """
    transform=transform.pipeline if isinstance(transform, DataTransform) else transform
    if stream_data:
        return DatasetIterable_(datalabel, dataset, transform, image_root=image_root)
    else:
        return Dataset_(datalabel, dataset, transform, oversample=oversample, image_root=image_root)



class Dataset_(torch.utils.data.Dataset):
    """A class to manupulate image data for training and inference
    
    Dataset is a class that generates a dataset for training or testing with PyTorch.
    It loads images from a directory (the subdirectories are recursively loaded),
    a list, a tuple, or a tab-separated (TSV) file.
    For the TSV file, the first column is recognized as the the path to the image
    and the second column as correct label if present.
    For traning, validation, and test, data should be input with TSV files containing two columns.

    Imbalanced data will make the model less sensitive to minority classes with small sample sizes
    compared to normal data for balanced data.
    Therefore, if models are created without properly addressing imbalanced data,
    problems will arise in terms of accuracy, computational complexity, etc.
    It is best to have balanced data during the data collection phase.
    However, if it is difficult to obtain balanced data in some situations,
    oversampling is used so that the samples in the minority class are equal in number to those in the major class.
    In this class, oversampling is performed by specifying `oversample=TRUE`.
    
    Args:
        datalabel: A DataLabel instance. This datalabel is used to convert class labels to integers.
        dataset: A path to a directory, a list, a tuple, or a TSV file.
        transform: A transform pipeline of image processing.
        balance_train: If True, the number of images in each class is balanced

    Examples:
        >>> from cvtk.ml import DataLabel
        >>> from cvtk.ml.torchutils import Dataset, DataTransform
        >>> 
        >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
        >>> 
        >>> transform = DataTransform(224, is_train=True)
        >>> 
        >>> dataset = Dataset(datalabel, 'train.txt', transform)
        >>> print(len(dataset))
        100
        >>> img, label = dataset[0]
        >>> print(img.shape)
        >>> print(label)
    """
    def __init__(self,
                 datalabel,
                 dataset: str|list|tuple,
                 transform: torchvision.transforms.Compose|DataTransform|None=None,
                 oversample: bool=False,
                 image_root: str|None=None):
        if transform is not None and isinstance(transform, DataTransform):
            transform = transform.pipeline            
        self.transform = transform
        self.oversample = oversample
        self.image_root = image_root
        self.x , self.y = self.__load_images(dataset, datalabel)
        if len(self.x) == 0:
            raise ValueError('No images are loaded. Check the dataset.')


    def __load_images(self, dataset, datalabel):
        x = []
        y = []
        if isinstance(dataset, str):
            if os.path.isfile(dataset):
                # load a single image, or images from a tab-separated file
                if filetype.is_image(dataset):
                    # load a single image file
                    x = [dataset]
                    y = [None]
                else:
                    # load a tab-separated file
                    if self.image_root is None:
                        image_base = os.path.dirname(os.path.abspath(dataset))
                    else:
                        image_base = os.path.abspath(self.image_root)
                    
                    if dataset.endswith('.gz') or dataset.endswith('.gzip'):
                        trainfh = gzip.open(dataset, 'rt')
                    else:
                        trainfh = open(dataset, 'r')
                    x = []
                    y = []
                    for data in trainfh:
                        data = data.rstrip().split('\t')
                        if len(data) == 0 or data[0] == '':
                            continue
                        
                        x.append(_resolve_image_path(data[0], image_base))
                        # set label to None if the file does not contain the label column in the second column

                        if len(data) >= 2 and data[1] != '':
                            y.append(datalabel[data[1]])
                        else:
                            y.append(None)
                    trainfh.close()
            elif os.path.isdir(dataset):
                # load images from a directory without labels
                for root, dirs, files in os.walk(dataset):
                    for f in files:
                        if filetype.is_image(os.path.join(root, f)):
                            x.append(os.path.join(root, f))
                            y.append(None)
        
        elif isinstance(dataset, list) or isinstance(dataset, tuple):
            # load images from a list or tuple
            for d in dataset:
                if isinstance(d, list) or isinstance(d, tuple):
                    if len(d) >= 2 and d[1] is not None and d[1] != '':
                        x.append(d[0])
                        y.append(datalabel[d[1]])
                    else:
                        x.append(d[0])
                        y.append(None)
                else:
                    x.append(d)
                    y.append(None)

        if self.oversample:
            x, y = self.__unbiased_classes(x, y, datalabel)

        return x, y


    def __getitem__(self, i):
        img = PIL.Image.open(self.x[i]).convert('RGB')
        img = PIL.ImageOps.exif_transpose(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.y[i] is None:
            return img
        else:
            return img, self.y[i]


    def __len__(self):
        return len(self.x)


    def __unbiased_classes(self, x, y, datalabel):
        n_images = [[] for _ in range(len(datalabel))]
        for i in range(len(y)):
            if y[i] is not None:
                n_images[y[i]].append(i)
            else:
                raise ValueError('The label is None. Oversample is only available for training data with labels.')

        # print errors when the input data does not allow oversampling
        empty_classes = [i for i, n in enumerate(n_images) if len(n) == 0]
        if len(empty_classes) > 0:
            missing_labels = [datalabel.labels[i] for i in empty_classes]
            raise ValueError(
                'Oversample requires all classes to be present at least once. '
                f'Missing classes: {missing_labels}'
            )

        n_images_max = max([len(n) for n in n_images])
        for i in range(len(n_images)):
            if len(n_images[i]) < n_images_max:
                n_images_sampled = random.choices(n_images[i], k=n_images_max - len(n_images[i]))
                x.extend([x[i] for i in n_images_sampled])
                y.extend([y[i] for i in n_images_sampled])

        return x, y



class DatasetIterable_(torch.utils.data.IterableDataset):
    def __init__(self,
                 datalabel,
                 dataset,
                 transform=None,
                 image_root=None):
        self.dataset = dataset
        self.datalabel = datalabel
        self.transform = transform
        self.n = self._calc_n_samples()
        
        if image_root is None:
            self.image_root = os.path.dirname(os.path.abspath(dataset))
        else:
            self.image_root = os.path.abspath(image_root)
    
    
    def _valid_image_record(self, data):
        data = data.rstrip()
        
        if data == '':
            return False
        
        data_records = data.split('\t')
        if len(data_records) == 0 or data_records[0] == '':
            return False
        
        return True
    
    
    def _open_file(self):
        if self.dataset.endswith(('.gz', '.gzip')):
            fh = gzip.open(self.dataset, 'rt')
        else:
            fh = open(self.dataset, 'r')
        return fh
    
    
    def _calc_n_samples(self):
        fh = self._open_file()
        n = 0
        for line in fh:
            if not self._valid_image_record(line):
                continue
            n += 1
        fh.close()
        return n
    

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start = 0
            step = 1
        else:
            start = worker_info.id
            step = worker_info.num_workers
        
        with self._open_file() as fh:
            for i, data in enumerate(fh):
                if i % step != start:
                    continue
                
                if not self._valid_image_record(data):
                    continue
                
                data_records = data.rstrip().split('\t')

                # image
                img_path = _resolve_image_path(data_records[0], self.image_root)
                img = PIL.Image.open(img_path).convert('RGB')
                img = PIL.ImageOps.exif_transpose(img)
                if self.transform is not None:
                    img = self.transform(img)

                # label
                if len(data_records) >= 2 and data_records[1] != '':
                    label = self.datalabel[data_records[1]]
                else:
                    label = None

                if label is None:
                    yield img
                else:
                    yield img, label
    
    
    def __len__(self):
        return self.n



class DataLoader(torch.utils.data.DataLoader):
    """DataLoader for managing image classification datasets.

    Wrapper around torch.utils.data.DataLoader with sensible defaults for image classification.
    Supports batching, shuffling, and parallel data loading via worker processes.

    Args:
        dataset: A PyTorch dataset object (typically from Dataset factory function).
        batch_size (int): Number of samples per batch. Default is 1.
        shuffle (bool): If True, shuffle data at every epoch. Default is False.
        num_workers (int): Number of worker processes for loading. Default is 0.
        **kwargs: Additional arguments passed to torch.utils.data.DataLoader.

    Examples:
        >>> from cvtk.ml import DataLabel
        >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader
        >>> datalabel = DataLabel(['cat', 'dog', 'bird'])
        >>> transform = DataTransform(224, is_train=True)
        >>> dataset = Dataset(datalabel, 'train.txt', transform)
        >>> dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class BaseRunner():
    """Base class for PyTorch model runners.

    Abstract base class providing common utilities for training and inference runners.
    Handles device initialization, data label management, workspace setup, and checkpoint management.
    Task-specific subclasses (e.g., ClsRunner) implement training, evaluation, and inference logic.
    """

    def __init__(self, datalabel, workspace=None, device='auto'):
        """Initialize the base runner.
        
        Args:
            datalabel (DataLabel|str|list|tuple): Class labels. Can be a DataLabel instance,
                file path, or list/tuple of label names.
            workspace (str|None): Directory for saving checkpoints and logs.
                If None, intermediate results are not persisted. Default is None.
            device (str): Device to run model on ('auto', 'cuda', 'cpu').
                'auto' automatically selects cuda if available. Default is 'auto'.
        """
        self.device = self._init_device(device)
        self.datalabel = self._init_datalabel(datalabel)
        self.workspace = self._init_tempdir(workspace)
        self.train_stats = None
        self.test_stats = None
        self.model = None

    def __del__(self):
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
        except:
            pass

    def _init_device(self, device='auto'):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _init_datalabel(self, datalabel):
        if isinstance(datalabel, DataLabel):
            return datalabel

        if isinstance(datalabel, str) or isinstance(datalabel, list) or isinstance(datalabel, tuple):
            return DataLabel(datalabel)

        raise TypeError('Invalid datalabel type: {}'.format(type(datalabel)))

    def _init_tempdir(self, workspace):
        if (workspace is not None) and (not os.path.exists(workspace)):
            os.makedirs(workspace)
        return workspace

    def _resolve_torchvision_attr(self, attr_name):
        obj = torchvision
        for attr in attr_name.split('.'):
            obj = getattr(obj, attr)
        return obj

    def _resolve_torchvision_model_attr(self, arch_name):
        obj = torchvision.models
        for attr in arch_name.split('.'):
            obj = getattr(obj, attr)
        return obj

    def _load_state_dict(self, model, weights, strict=False):
        state_dict = torch.load(weights, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
        return model

    def _checkpoint_datalabel_path(self, weights):
        return os.path.splitext(weights)[0] + '.dl.txt'

    def _str(self, s):
        if s is None:
            return 'NA'
        return str(s)

    def _ensure_output_dir(self, output):
        outdir = os.path.dirname(output)
        if outdir != '' and not os.path.exists(outdir):
            os.makedirs(outdir)

    def _save_datalabel(self, output):
        self.datalabel.save(os.path.splitext(output)[0] + '.dl.txt')

    def _move_model_to_cpu_for_save(self):
        if self.model is not None:
            self.model = self.model.to('cpu')

    def _restore_model_to_device_after_save(self):
        if self.model is not None:
            self.model = self.model.to(self.device)






class ClsRunner(BaseRunner):
    """Classification model runner for training and inference with PyTorch.

    High-level interface for image classification training, evaluation, and inference.
    Supports model selection from torchvision, weight loading, mixed precision training,
    and various output formats. Automatically handles checkpoint management and logging.

    Args:
        datalabel (str|list|tuple|DataLabel): Class labels. If string (file path), list, or tuple,
            converted to DataLabel instance.
        model (str|torch.nn.Module): Torchvision model name (e.g., 'efficientnet_b7') or
            torch.nn.Module instance.
        weights (str|None): Path to pretrained weights or torchvision weights enum
            (e.g., 'EfficientNet_B7_Weights.DEFAULT'). Default is None.
        workspace (str|None): Directory for saving checkpoints and logs.
            If None, intermediate results not persisted. Default is None.
        device (str): Device to run on ('auto', 'cuda', 'cpu'). Default is 'auto'.

    Attributes:
        device (str): Actual device used ('cuda' or 'cpu').
        datalabel (DataLabel): Class label manager.
        model (torch.nn.Module): The classification model.
        workspace (str|None): Checkpoint/log directory.
        train_stats (dict): Training statistics (epoch, loss, accuracy).
        test_stats (dict): Test statistics (loss, accuracy, scores).

    Examples:
        >>> from cvtk.ml.torchutils import ClsRunner
        >>> datalabel = ['cat', 'dog', 'bird']
        >>> runner = ClsRunner(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
    """
    def __init__(self, datalabel, model, weights=None, workspace=None, device='auto'):
        """Initialize classification runner.
        
        Args:
            datalabel: Class labels (DataLabel, filepath, list, or tuple).
            model: Torchvision model name or torch.nn.Module instance.
            weights: Pretrained weights path or torchvision weights enum. Default is None.
            workspace: Checkpoint/log directory. Default is None.
            device: Device to use ('auto', 'cuda', 'cpu'). Default is 'auto'.
        """
        super().__init__(datalabel=datalabel, workspace=workspace, device=device)

        self.task_type = 'cls'
        self.model = self.__init_model(model, weights, len(self.datalabel.labels))
        self.model = self.model.to(self.device)


    def __build_amp_scaler(self):
        if self.device != 'cuda':
            return None
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            return torch.amp.GradScaler()
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
            return torch.cuda.amp.GradScaler()
        return None


    def __autocast_context(self):
        if self.device != 'cuda':
            return contextlib.nullcontext()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast(device_type='cuda', enabled=True)
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            return torch.cuda.amp.autocast(enabled=True)
        return contextlib.nullcontext()
    

    def __init_model(self, model, weights, n_classes):
        

        # fix the output layer
        def __set_output(module, n_classes):
            last_layer_name = None
            last_layer = None

            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    last_layer_name = name
                    last_layer = child
                else:
                    sub_last_layer_name, sub_last_layer = __set_output(child, n_classes)
                    if sub_last_layer:
                        last_layer_name = f'{name}.{sub_last_layer_name}'
                        last_layer = sub_last_layer

            if last_layer:
                in_features = last_layer.in_features
                new_layer = torch.nn.Linear(in_features, n_classes)
                layers = last_layer_name.split('.')
                sub_module = module
                for layer in layers[:-1]:
                    sub_module = getattr(sub_module, layer)
                setattr(sub_module, layers[-1], new_layer)

            return last_layer_name, last_layer
        
        
        module = None
        
        if isinstance(model, str):
            try:
                model_builder = self._resolve_torchvision_model_attr(model)
            except AttributeError as exc:
                raise ValueError(f'Unknown torchvision model: {model}') from exc
            
            if not callable(model_builder):
                raise ValueError(f'Invalid torchvision model: {model}')

            if weights is None:
                module = model_builder(weights=None)
            elif isinstance(weights, str):
                if os.path.exists(weights):
                    module = model_builder(weights=None) # setup weights later
                else:
                    try:
                        resolved_weights = self._resolve_torchvision_model_attr(weights) #  e.g. EfficientNet_B7_Weights.DEFAULT
                    except AttributeError as exc:
                        raise ValueError(
                            f'Unknown torchvision weights: {weights}. '
                            f'Use a valid enum path like EfficientNet_B7_Weights.DEFAULT '
                            f'or provide a local checkpoint path.'
                        ) from exc
                    module = model_builder(weights=resolved_weights)
            else:
                module = model_builder(weights=weights)

        elif isinstance(model, torch.nn.Module):
            module = model
        
        else:
            raise ValueError('Invalid model type: {}. Use a torchvision model name or torch.nn.Module.'.format(type(model)))

        
        # load weights
        # As the weights may have been pre-trained with different number of classes,
        # fix the output layer matching the number of classes during loading weights,
        # and then fix it to match the number of classes of the current datalabel if necessary.
        datalabel_loaded = None
        if weights is not None and os.path.exists(weights):
            dl_path = os.path.splitext(weights)[0] + '.dl.txt'

            if os.path.exists(dl_path):
                datalabel_loaded = DataLabel(dl_path)
                __set_output(module, len(datalabel_loaded))
                 
            state_dict = torch.load(weights, map_location='cpu')
            module.load_state_dict(state_dict, strict=False)
        
            if (datalabel_loaded is None) or (len(datalabel_loaded.labels) != n_classes):
                __set_output(module, n_classes)
                print('n_classes is finally changed to '+ str(n_classes))
        
        else:
            __set_output(module, n_classes)
        
        return module
    



    def train(self, train, valid=None, test=None, epoch=20, optimizer='auto', criterion='auto', scaler='auto', resume=False):
        """Train the model with provided dataloaders.

        Trains the model for specified epochs with optional validation and testing.
        Training statistics are logged and saved to workspace if provided.
        Supports mixed precision training (AMP) when available and resumable checkpointing.

        Args:
            train (torch.utils.data.DataLoader): DataLoader for training data.
            valid (torch.utils.data.DataLoader|None): DataLoader for validation. Default is None.
            test (torch.utils.data.DataLoader|None): DataLoader for testing at end of training. Default is None.
            epoch (int): Number of epochs to train. Default is 20.
            optimizer (torch.optim.Optimizer|str): Optimizer instance or 'auto' for SGD. Default is 'auto'.
            criterion (torch.nn.Module|str): Loss function or 'auto' for CrossEntropyLoss. Default is 'auto'.
            scaler (torch.amp.GradScaler|str|None): Gradient scaler for AMP or 'auto' for automatic selection. Default is 'auto'.
            resume (bool): If True, resume from last checkpoint in workspace. Default is False.
        
        Returns:
            None. Training stats saved to train_stats attribute and workspace if provided.
        
        Examples:
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, ClsRunner
            >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
            >>> model = ClsRunner(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
            >>> transform_train = DataTransform(600, is_train=True)
            >>> dataset_train = Dataset(datalabel, 'train.txt', transform_train)
            >>> dataloader_train = DataLoader(dataset_train, batch_size=32, num_workers=4)
            >>> model.train(dataloader_train, epoch=20)
        """

        self.train_stats = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': []
        }

        dataloaders = {'train': train, 'valid': valid, 'test': test}

        # training params
        criterion = torch.nn.CrossEntropyLoss() if criterion == 'auto' else criterion
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3) if optimizer == 'auto' else optimizer
        scaler = self.__build_amp_scaler() if scaler == 'auto' else scaler
        
        # resume training from the last checkpoint if resume is True
        last_epoch = 0
        if resume:
            last_epoch = self.__update_model_weight()

        # train the model
        for epoch_i in range(last_epoch + 1, epoch + 1):
            print(f'Epoch {epoch_i}/{epoch}')

            # training and validation
            self.train_stats['epoch'].append(epoch_i)
            for phase in ['train', 'valid']:
                loss, acc, probs = self.__train(dataloaders[phase], phase, criterion, optimizer, scaler)
                self.train_stats[f'{phase}_loss'].append(loss)
                self.train_stats[f'{phase}_acc'].append(acc)
                if loss is not None and acc is not None:
                    print(f'{phase} loss: {loss:.4f}, acc: {acc:.4f}')

            # test the model if dataset is provided at the last epoch
            if epoch_i == epoch and dataloaders['test'] is not None:
                self.test(dataloaders['test'], criterion)
            
            if self.workspace is not None:
                self.save(os.path.join(self.workspace, f'checkpoint_latest.pth'))


    def __update_model_weight(self):
        last_epoch = 0
        if self.workspace is None:
            return last_epoch

        trainstats_fpath = os.path.join(self.workspace, 'checkpoint_latest.train_stats.txt')
        chk_fpath = os.path.join(self.workspace, 'checkpoint_latest.pth')
        if os.path.exists(trainstats_fpath) and os.path.exists(chk_fpath):
            # update train stats
            with open(trainstats_fpath, 'r') as fh:
                tags = fh.readline().strip().split('\t')
                for tag in tags:
                    self.train_stats[tag] = []
                for f_line in fh:
                    vals = f_line.strip().split('\t')
                    for tag, val in zip(tags, vals):
                        if val in ('NA', 'None', ''):
                            val = None
                        else:
                            if tag == 'epoch':
                                val = int(val)
                            else:
                                val = float(val)
                        self.train_stats[tag].append(val)
            # update model weight with the last checkpoint
            self.model = self.model.to('cpu')
            self._load_state_dict(self.model, chk_fpath, strict=True)
            self.model = self.model.to(self.device)
            last_epoch = max(self.train_stats['epoch'])
            
        return last_epoch


    def __train(self, dataloader, phase, criterion, optimizer, scaler, score_type='softmax'):
        if dataloader is None:
            return None, None, None
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        n_samples = 0
        scores = []

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if phase == 'train':
                optimizer.zero_grad(set_to_none=True)
                
            with torch.set_grad_enabled(phase == 'train'):
                if scaler is None:
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                else:
                    if self.device == 'cuda':
                        with self.__autocast_context():
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            running_loss += loss.detach().item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            n_samples += inputs.size(0)
            scores.append(self.__format_scores(outputs, score_type))

        if n_samples == 0:
            raise ValueError('No samples were processed from dataloader.')

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double().item() / n_samples
        scores = np.concatenate(scores, axis=0).tolist()
        return epoch_loss, epoch_acc, scores


    def __format_scores(self, scores, score_type):
        if scores.ndim != 2:
            raise ValueError('The score have shape of (batch_size, n_classes), but got {}'.format(scores.shape))

        scores = scores.detach().float()
        score_type = score_type.lower()
        if score_type in ['logit', 'logits']:
            pass        
        elif score_type == 'softmax':
            scores = torch.softmax(scores, dim=1)
        elif score_type == 'sigmoid':
            scores = torch.sigmoid(scores)
        else:
            raise ValueError(f"Unsupported score_type: {score_type}. Use 'logits', 'softmax', or 'sigmoid'.")

        return scores.cpu().numpy()



    def save(self, output):
        """Save model weights, data labels, and training logs.

        Saves the trained model weights and associated metadata. Creates output directory if needed.
        Also saves training statistics and test outputs as separate files with same base name.
        The model and datalabel are saved as a pair for later loading.

        Args:
            output (str): File path for model weights. Auto-appends '.pth' if missing.

        Returns:
            None. Files saved:
                - {output}.pth: Model weights
                - {output}.dl.txt: DataLabel (class names)
                - {output}.train_stats.txt: Training statistics
                - {output}.test_outputs.txt: Test outputs (if available)

        Examples:
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import ClsRunner
            >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
            >>> model = ClsRunner(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
            >>> # ... training ...
            >>> model.save('output/plant_classifier.pth')
        """
        if not output.endswith('.pth'):
            output += '.pth'

        self._ensure_output_dir(output)

        self._move_model_to_cpu_for_save()
        torch.save(self.model.state_dict(), output)
        self._restore_model_to_device_after_save()

        self._save_datalabel(output)
        
        

        output_log_fpath = os.path.splitext(output)[0] + '.train_stats.txt'
        self.__write_train_stats(output_log_fpath)

        if self.test_stats is not None:
            output_log_fpath = os.path.splitext(output)[0] + '.test_outputs.txt'
            self.__write_test_outputs(output_log_fpath)


    def __write_train_stats(self, output_log_fpath):
        with open(output_log_fpath, 'w') as fh:
            fh.write('\t'.join(self.train_stats.keys()) + '\n')
            for vals in zip(*self.train_stats.values()):
                fh.write('\t'.join([self._str(v) for v in vals]) + '\n')


    

    def __write_test_outputs(self, output_log_fpath):
        with open(output_log_fpath, 'w') as fh:
            fh.write('# loss: {}\n'.format(self.test_stats['loss']))
            fh.write('# acc: {}\n'.format(self.test_stats['acc']))
            fh.write('\t'.join(['image', 'label'] + self.datalabel.labels) + '\n')

            dataset = self.test_stats.get('dataset')
            scores = self.test_stats.get('scores', [])
            images = getattr(dataset, 'x', None)
            labels = getattr(dataset, 'y', None)
            if not isinstance(images, (list, tuple)) or len(images) != len(scores):
                images = [f'sample_{i:06d}' for i in range(len(scores))]
            if not isinstance(labels, (list, tuple)) or len(labels) != len(scores):
                labels = [None for _ in range(len(scores))]

            for x_, y_, p_ in zip(images, labels, scores):
                if isinstance(y_, (int, np.integer)) and 0 <= y_ < len(self.datalabel.labels):
                    label_str = self.datalabel.labels[y_]
                elif y_ is None:
                    label_str = 'NA'
                else:
                    label_str = str(y_)
                fh.write('{}\t{}\t{}\n'.format(
                    x_,
                    label_str,
                    '\t'.join([str(_) for _ in p_])))
                

    def test(self, dataloader, criterion=None, score_type='softmax'):
        """Evaluate model on test dataset.
        
        Runs model in evaluation mode on test data and computes loss, accuracy, and class scores.
        Results are stored in test_stats for later access or saving.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for test data.
            criterion (torch.nn.Module|None): Loss function. Default is None (uses CrossEntropyLoss).
            score_type (str): Score format: 'logits', 'softmax', or 'sigmoid'. Default is 'softmax'.
        
        Returns:
            dict: Test statistics containing:
                - 'dataset': The dataset object
                - 'loss': Average loss on test data
                - 'acc': Accuracy on test data
                - 'scores': Predicted scores/probabilities for each sample
        
        Examples:
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, ClsRunner
            >>> datalabel = DataLabel(['cat', 'dog', 'bird'])
            >>> model = ClsRunner(datalabel, 'efficientnet_b7', 'model.pth')
            >>> transform = DataTransform(224, is_train=False)
            >>> dataset = Dataset(datalabel, 'test.txt', transform)
            >>> dataloader = DataLoader(dataset, batch_size=32)
            >>> stats = model.test(dataloader)
            >>> print(f"Test accuracy: {stats['acc']:.4f}")
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
        loss, acc, scores = self.__train(dataloader, 'test', criterion, None, None, score_type)
        self.test_stats = {
                    'dataset': dataloader.dataset,
                    'loss': loss,
                    'acc': acc,
                    'scores': scores
                }
        return self.test_stats


    def inference(self, data, format='pandas', batch_size=32, num_workers=8, score_type='softmax'):
        """Perform inference on images with the trained model.

        Runs the model in evaluation mode on input images and returns predictions in requested format.
        Handles various input types: DataLoader, image list, single image, or directory.
        Automatically uses appropriate image size from training.

        Args:
            data (torch.utils.data.DataLoader|str|list): Input data:
                - DataLoader: PyTorch DataLoader with test data
                - str: File path to single image, TSV file, or directory of images
                - list: List of image paths with optional labels
            format (str): Output format: 'pandas' (DataFrame), 'list', 'dict', 'numpy', or 'np'. Default is 'pandas'.
            batch_size (int): Batch size for inference. Default is 32.
            num_workers (int): Number of workers for data loading. Default is 8.
            score_type (str): Score format: 'logits', 'softmax', or 'sigmoid'. Default is 'softmax'.
        
        Returns:
            Predictions in requested format:
            - 'pandas': DataFrame with columns as class names, rows as images
            - 'list': List of score lists, one per image
            - 'dict': List of dicts with 'label' and 'score' keys per class
            - 'numpy'/'np': numpy array of shape (n_samples, n_classes)
        
        Examples:
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, ClsRunner
            >>> datalabel = DataLabel(['cat', 'dog', 'bird'])
            >>> model = ClsRunner(datalabel, 'efficientnet_b7', 'model.pth')
            >>> transform = DataTransform(224, is_train=False)
            >>> dataset = Dataset(datalabel, 'test_images/', transform)
            >>> dataloader = DataLoader(dataset, batch_size=32)
            >>> probs = model.inference(dataloader, format='pandas')
            >>> probs.to_csv('predictions.txt', sep='\\t')
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        if isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            dataloader = DataLoader(
                Dataset(self.datalabel, data, transform=DataTransform(512, is_train=False)),
                batch_size=batch_size, num_workers=num_workers)

        scores = []
        for inputs in dataloader:
            if not isinstance(inputs, torch.Tensor):
                inputs = inputs[0]
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
            
            scores.append(self.__format_scores(outputs, score_type))
        labels = self.datalabel.labels
        if len(scores) == 0:
            empty_scores = np.empty((0, len(labels)), dtype=float)
            return self.__format_inference_output(empty_scores, labels, [], format)

        scores = np.concatenate(scores, axis=0)
        labels = self.datalabel.labels

        dataset_obj = getattr(dataloader, 'dataset', None)
        dataset_images = getattr(dataset_obj, 'x', None)

        if isinstance(dataset_images, (list, tuple)) and len(dataset_images) == scores.shape[0]:
            images = list(dataset_images)
        else:
            # fallback for generic PyTorch datasets that do not expose image paths.
            images = [f'sample_{i:06d}' for i in range(scores.shape[0])]

        return self.__format_inference_output(scores, labels, images, format)



    def __format_inference_output(self, scores, labels, images, format):
        format = format.lower()
        
        if format in ['np', 'numpy', 'array']:
            outputs = scores
            
        elif format in ['list', 'tuple']:
            outputs = []
            for i in range(scores.shape[0]):
                outputs.append(scores[i].tolist())
                
        elif format == 'dict':
            outputs = []
            for i in range(scores.shape[0]):
                output_ = []
                for l_, p_ in zip(labels, scores[i].tolist()):
                    output_.append({'label': l_, 'score': p_})
                outputs.append(output_)
            
        else:
            outputs = pd.DataFrame(scores, columns=labels, index=images)
            
        return outputs

