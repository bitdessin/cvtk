import os
import random
import gzip
import importlib
import filetype
import numpy as np
import pandas as pd
import sklearn.metrics
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import PIL.ImageFilter
import torch
import torchvision
import torchvision.transforms.v2
from cvtk.ml.data import DataLabel
from ._subutils import __get_imports, __insert_imports, __extend_cvtk_imports, __del_docstring



def DataTransform(shape, is_train=False):
    """Generate image preprocessing pipeline

    DataTransforms is a function to generate image preprocessing pipeline used in PyTorch.
    By default (`is_train=False`), a pipeline for inference is generated,
    which resizes images to a square shape and converts them to a tensor.
    A pipeline for training is generated when `is_train=True`,
    in which images are resized to a square shape with a specified resolution,
    then processed with several fundamental augmentation processes, and finally converted to a tensor.
            
    Args:
        shape (int): The resolution of the square image.
        is_train (bool): If True, a pipeline for training is generated. Default is False.

    Returns:
        A torchvision.transforms.Compose instance containing the transform pipeline. 
   
    Examples:
        >>> from cvtk.ml.torchutils import DataTransform
        >>> 
        >>> transform_train = DataTransform(224, is_train=True)
        >>> print(transform_train)
        >>>
        >>> transform_inference = DataTransform(224)
        >>> print(transforms_inference)
    """
    if is_train:
        return torchvision.transforms.Compose([
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.Resize(size=(shape + 50, shape + 50), antialias=True),
                torchvision.transforms.v2.RandomResizedCrop(size=(shape, shape), antialias=True),
                torchvision.transforms.v2.RandomHorizontalFlip(0.5),
                torchvision.transforms.v2.RandomAffine(45),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    else:
        return torchvision.transforms.Compose([
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.Resize(size=(shape, shape), antialias=True),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])



class Dataset(torch.utils.data.Dataset):
    """Generate dataset for training or testing with PyTorch 

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
    upsampling is used so that the samples in the minority class are equal in number to those in the major class.
    In this class, upsampling is performed by specifying `upsampling=TRUE`.
    
    Args:
        datalabel (DataLabel): A DataLabel instance. This datalabel is used to convert class labels to integers.
        dataset (str|list|tuple): A path to a directory, a list, a tuple, or a TSV file.
        transform (None|torchvision.transforms.Compose): A transform pipeline of image processing.
        balance_train (bool): If True, the number of images in each class is balanced

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
                 dataset,
                 transform=None,
                 upsampling=False):
        self.transform = transform
        self.upsampling = upsampling
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
                    if dataset.endswith('.gz') or dataset.endswith('.gzip'):
                        trainfh = gzip.open(dataset, 'rt')
                    else:
                        trainfh = open(dataset, 'r')
                    x = []
                    y = []
                    for line in trainfh:
                        words = line.rstrip().split('\t')
                        x.append(words[0])
                        # set label to None if the file does not contain the label column in the second column
                        if len(words) >= 2:
                            y.append(datalabel[words[1]])
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
                    if len(d) >= 2:
                        x.append(d[0])
                        y.append(datalabel[d[1]])
                    else:
                        x.append(d[0])
                        y.append(None)
                else:
                    x.append(d)
                    y.append(None)

        if self.upsampling:
            x, y = self.__unbiased_classes(x, y)

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


    def __unbiased_classes(self, x, y):
        n_images = [[]] * len(self.datalabel)
        for i in range(len(y)):
            n_images[y[i]].append(i)

        n_images_max = max([len(n) for n in n_images])
        for i in range(len(n_images)):
            if len(n_images[i]) < n_images_max:
                n_images_sampled = random.choices(n_images[i], k=n_images_max - len(n_images[i]))
                x.extend([x[i] for i in n_images_sampled])
                y.extend([y[i] for i in n_images_sampled])

        return x, y



def DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False):
    """Create dataloader to manage data for training and inference

    This function simply creates a torch.utils.data.DataLoader instance to manage data for training and inference.

    Args:
        dataset (cvtk.ml.torchutils.DataSet): A dataset for training and inference.
        batch_size (int): A batch size for training and inference.
        num_workers (int): The number of workers for data loading.
        shuffle (bool): If True, the data is shuffled at every epoch.

    Returns:
        A torch.utils.data.DataLoader instance.

    Examples:
        >>> from cvtk.ml
        >>> from cvtk.ml import DataLabel
        >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader
        >>>
        >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
        >>> transform = DataTransform(224, is_train=True)
        >>> dataset = Dataset(datalabel, 'train.txt', transform)
        >>> dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>> 
    """
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)



class CLSCORE():
    """A class provides training and inference functions for a classification model using PyTorch

    CLSCORE is a class that provides training and inference functions for a classification model.

    Args:
        datalabel (str|list|tuple|DataLabel): A DataLabel instance containing class labels.
            If string (of file path), list, tuple is given, it is converted to a DataLabel instance.
        model (str|torch.nn.Module): A string to specify a model or a torch.nn.Module instance.
        weights (str): A file path to model weights.
        workspace (str): A temporary directory path to save intermediate checkpoints and training logs.
            If not given, the intermediate results are not saved.

    Attributes:
        device (str): A device to run the model. Default is 'cuda' if available, otherwise 'cpu'.
        datalabel (DataLabel): A DataLabel instance containing class labels.
        model (torch.nn.Module): A model of torch.nn.Module instance.
        workspace (str): A temporary directory path.
        train_stats (dict): A dictionary to save training statistics
        test_stats (dict): A dictionary to save test statistics

    Examples:
        >>> import torch
        >>> import torchvision
        >>> from cvtk.ml.torchutils import CLSCORE
        >>>
        >>> datalabel = ['leaf', 'flower', 'root']
        >>> m = CLSCORE(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
        >>> 
        >>> datalabel = 'class_label.txt'
        >>> m = CLSCORE(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
    """
    def __init__(self, datalabel, model, weights=None, workspace=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datalabel = self.__init_datalabel(datalabel)
        self.model = self.__init_model(model, weights, len(self.datalabel.labels))
        self.workspace = self.__init_tempdir(workspace)
        
        self.model = self.model.to(self.device)
        
        self.train_stats = None
        self.test_stats = None

    
    def __init_datalabel(self, datalabel):
        if isinstance(datalabel, DataLabel):
            pass
        if isinstance(datalabel, str) or isinstance(datalabel, list) or isinstance(datalabel, tuple):
            datalabel = DataLabel(datalabel)
        elif not isinstance(datalabel, DataLabel):
            raise TypeError('Invalid type: {}'.format(type(datalabel)))
        return datalabel


    def __init_model(self, model, weights, n_classes):
        if isinstance(model, str):
            if weights is None:
                module = eval(f'torchvision.models.{model}(weights=None)')
            else:
                if os.path.exists(weights):
                    module = eval(f'torchvision.models.{model}(weights=None)')
                else:
                    module = eval(f'torchvision.models.{model}(weights=torchvision.models.{weights})')
        
        elif isinstance(model, str):
            is_torch_model = False
            for name in dir(torchvision.models):
                obj = getattr(torchvision.models, name)
                if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                    if isinstance(model, obj):
                        is_torch_model = True
                        break
            if is_torch_model:
                module = model
        
        elif isinstance(model, torch.nn.Module):
            module = model
        
        else:
            raise ValueError('Invalid model type: {}'.format(type(model)))



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

        __set_output(module, n_classes)

        if weights is not None and os.path.exists(weights):
            module.load_state_dict(torch.load(weights))
        
        return module
    


    def __init_tempdir(self, workspace):
        if (workspace is not None) and (not os.path.exists(workspace)):
            os.makedirs(workspace)
        return workspace



    def train(self, train, valid=None, test=None, epoch=20, optimizer=None, criterion=None, resume=False):
        """Train the model with the provided dataloaders

        Train the model with the provided dataloaders. The training statistics are saved in the temporary directory.

        Args:
            train (torch.utils.data.DataLoader): A dataloader for training.
            valid (torch.utils.data.DataLoader): A dataloader for validation.
            test (torch.utils.data.DataLoader): A dataloader for testing.
            epoch (int): The number of epochs to train the model.
            optimizer (torch.optim.Optimizer|None): An optimizer for training.
                Default is `None` and `torch.optim.SGD` is used.
            criterion (torch.nn.Module|None): A loss function for training.
                Default is `None` and `torch.nn.CrossEntropyLoss` is used.
            resume (bool): If True, the training resumes from the last checkpoint
                which is saved in the temporary directory specified with ``temp_dirpath``.
        
        Examples:
            >>> import torch
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, CLSCORE
            >>> 
            >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
            >>> 
            >>> model = CLSCORE(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
            >>>
            >>> # train dataset
            >>> transforms_train = DataTransform(600, is_train=True)
            >>> dataset_train = Dataset(datalabel, 'train.txt', transforms_train)
            >>> dataloader_train = DataLoaders(dataset_train, batch_size=32, num_workers=4)
            >>> # valid dataset
            >>> transforms_valid = DataTransform(600, is_train=False)
            >>> dataset_valid = Dataset(datalabel, 'valid.txt, transforms_valid)
            >>> dataloader_valid = DataLoader(dataset_valid, batch_size=32, num_workers=4)
            >>>
            >>> model.train(dataloader_train, dataloader_valid, epoch=20)
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
        criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3) if optimizer is None else optimizer
        
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
                loss, acc, probs = self.__train(dataloaders[phase], phase, criterion, optimizer)
                self.train_stats[f'{phase}_loss'].append(loss)
                self.train_stats[f'{phase}_acc'].append(acc)
                if loss is not None and acc is not None:
                    print(f'{phase} loss: {loss:.4f}, acc: {acc:.4f}')

            # test the model if dataset is provided at the last epoch
            if epoch_i == epoch and dataloaders['test'] is not None:
                loss, acc, probs = self.__train(dataloaders['test'], phase, criterion, optimizer)
                self.test_stats = {
                    'dataset': dataloaders['test'].dataset,
                    'loss': loss,
                    'acc': acc,
                    'probs': probs
                }
            
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
                        if val is not None:
                            if val != 'NA' and val != 'None':
                                if tag == 'epoch':
                                    val = int(val)
                                else:
                                    val = float(val)
                        self.train_stats[tag].append(val)
            # update model weight with the last checkpoint
            self.model = self.model.to('cpu')
            self.model.load_state_dict(torch.load(chk_fpath))
            self.model = self.model.to(self.device)
            last_epoch = max(self.train_stats['epoch'])
            
        return last_epoch


    def __train(self, dataloader, phase, criterion, optimizer):
        if dataloader is None:
            return None, None, None
        if phase == 'trian':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        probs = []

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            probs.append(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

        epoch_loss = running_loss.double().item() / len(dataloader.dataset)
        epoch_acc = running_corrects.double().item() / len(dataloader.dataset)
        probs = np.concatenate(probs, axis=0).tolist()
        return epoch_loss, epoch_acc, probs



    def save(self, output):
        """Save model weights and training logs

        Save model weights in a file specified with the `output` argument.
        The extension of the output file should be '.pth'; if not, '.pth' is appended to the output file path.
        Additionally, if training logs and test outputs are present,
        they are saved in text files with the same name as weights
        but with '.train_stats.txt' and '.test_outputs.txt' extensions, respectively.

        Args:
            output (str): A file path to save the model weights.

        Examples:
            >>> import torch
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, CLSCORE
            >>> 
            >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
            >>> model = CLSCORE(datalabel, 'efficientnet_b7', 'EfficientNet_B7_Weights.DEFAULT')
            >>> 
            >>> # training
            >>> # ...
            >>> model.save('output/plant_organ_classification.pth')
        """
        if not output.endswith('.pth'):
            output += '.pth'
        if not os.path.exists(os.path.dirname(output)):
            if os.path.dirname(output) != '':
                os.makedirs(os.path.dirname(output))

        self.model = self.model.to('cpu')
        
        torch.save(self.model.state_dict(), output)
        self.model = self.model.to(self.device)

        output_log_fpath = os.path.splitext(output)[0] + '.train_stats.txt'
        self.__write_train_stats(output_log_fpath)

        if self.test_stats is not None:
            output_log_fpath = os.path.splitext(output)[0] + '.test_outputs.txt'
            self.__write_test_outputs(output_log_fpath)


    def __write_train_stats(self, output_log_fpath):
        with open(output_log_fpath, 'w') as fh:
            fh.write('\t'.join(self.train_stats.keys()) + '\n')
            for vals in zip(*self.train_stats.values()):
                fh.write('\t'.join([self.__str(v) for v in vals]) + '\n')


    def __str(self, s):
        if s is None:
            return 'NA'
        return str(s)
    

    def __write_test_outputs(self, output_log_fpath):
        with open(output_log_fpath, 'w') as fh:
            fh.write('# loss: {}\n'.format(self.test_stats['loss']))
            fh.write('# acc: {}\n'.format(self.test_stats['acc']))
            fh.write('\t'.join(['image', 'label'] + self.datalabel.labels) + '\n')
            for x_, y_, p_ in zip(self.test_stats['dataset'].x, self.test_stats['dataset'].y, self.test_stats['probs']):
                fh.write('{}\t{}\t{}\n'.format(
                    x_,
                    self.datalabel.labels[y_],
                    '\t'.join([str(_) for _ in p_])))
                


    def inference(self, data, value='prob+label', format='pandas', batch_size=32, num_workers=8):
        """Perform inference with the input images

        Perform inference with the input images with the trained model.
        The format of ouput can be specified with `output` and `format` arguments.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader for inference.
            output (str): A string to specify the information of inference result for output.
                Probabilities ('prob'), labels ('label'), or both ('prob+label') can be specified.
            format (str): A string to specify output format in Pandas Data.Frame ('pandas'),
                NumPy array ('numpy'), list ('list'), or tuple ('tuple').
        
        Examples:
            >>> import torch
            >>> from cvtk.ml import DataLabel
            >>> from cvtk.ml.torchutils import DataTransform, Dataset, DataLoader, CLSCORE
            >>> 
            >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
            >>>
            >>> model = CLSCORE(datalabel, 'efficientnet_b7', 'plant_organs.pth')
            >>>
            >>> transform = DataTransform(600)
            >>> dataset = Dataset(datalabel, 'sample.jpg', transform)
            >>> dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
            >>> 
            >>> probs = model.inference(dataloader)
            >>> probs.to_csv('inference_results.txt', sep = '\t', header=True, index=True, index_label='image')
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        if isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            dataloader = DataLoader(
                Dataset(self.datalabel, data, transform=DataTransform(512, is_train=False)),
                batch_size=batch_size, num_workers=num_workers)

        probs = []
        for inputs in dataloader:
            if not isinstance(inputs, torch.Tensor):
                inputs = inputs[0]
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
            probs.append(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        labels = self.datalabel[probs.argmax(axis=1).tolist()]
        
        return self.__format_inference_output(probs, labels, dataloader.dataset.x, self.datalabel.labels, value, format)



    def __format_inference_output(self, probs, labels, images, cl, value, format):
        if value == 'prob':
            if format in ['list']:
                return probs.tolist()
            elif format in ['tuple']:
                return tuple(probs.tolist())
            elif format in ['numpy', 'np']:
                return probs
            else:
                return pd.DataFrame(probs, index=images, columns=cl)
        elif value == 'label':
            if format in ['list']:
                return labels
            elif format in ['tuple']:
                return tuple(labels)
            elif format in ['numpy', 'np']:
                raise ValueError('The inferenced labels cannot be converted to numpy array, use `list` or `tuple` instead.')
            else:
                return pd.DataFrame(labels, index=images, columns=['prediction'])    
        else:
            if format in ['list']:
                return list(zip(probs.tolist(), labels))
            elif format in ['tuple']:
                return tuple(zip(probs.tolist(), labels))
            elif format in ['numpy', 'np']:
                raise ValueError('The inferenced labels cannot be converted to numpy array, use `list` or `tuple` instead.')
            else:
                return pd.DataFrame(np.concatenate([np.array(labels).reshape(-1, 1), probs], axis=1),
                                    index=images, columns=['prediction'] + cl)



def plot_trainlog(train_log, output=None, title='Training Statistics', mode='lines', width=600, height=800, scale=1.0):
    """Plot training log

    Plot loss and accuracy at each epoch from the training log which
    is expected to be saved in a tab-separated file with the following format:

    ::

        epoch  train_loss  train_acc  valid_loss  valid_acc
        1      1.40679     0.22368    1.24780     0.41667
        2      1.21213     0.48684    1.09401     0.83334
        3      1.00425     0.81578    0.88967     0.83334
        4      0.78659     0.82894    0.64055     0.91666
        5      0.46396     0.96052    0.39010     0.91666

    
    Args:
        train_log (str): A path to a tab-separated file containing training logs.
        output (str): A file path to save the output images. If not provided, the plot is shown on display.
        width (int): A width of the output image.
        height (int): A height of the output image.
        scale (float): The scale of the output image, which is used to adjust the resolution.
    """
    # data preparation
    train_log = pd.read_csv(train_log, sep='\t', header=0, comment='#')
    train_log = train_log.melt(id_vars='epoch', var_name='type', value_name='value')
    train_log = train_log.assign(phase=train_log['type'].apply(lambda x: x.split('_')[0]))
    train_log = train_log.assign(metric=train_log['type'].apply(lambda x: x.split('_')[1]))
    
    # plots
    cols = px.colors.qualitative.Plotly
    fig = plotly.subplots.make_subplots(rows=2, cols=1)

    c = 0
    for phase in train_log['phase'].unique():
        d = train_log[(train_log['phase'] == phase) & (train_log['metric'] == 'loss')]
        fig.add_trace(
            go.Scatter(x=d['epoch'], y=d['value'], mode=mode, name=phase, line=dict(color=cols[c])),
            row=1, col=1)
        
        d = train_log[(train_log['phase'] == phase) & (train_log['metric'] == 'acc')]
        fig.add_trace(
            go.Scatter(x=d['epoch'], y=d['value'], mode=mode, name=phase, line=dict(color=cols[c]), showlegend=False),
            row=2, col=1)
        
        c = (c + 1) % len(cols)

    fig.update_layout(title_text=title, template='ggplot2')
    fig.update_xaxes(title_text='epoch')
    fig.update_yaxes(title_text='loss', row=1, col=1)
    fig.update_yaxes(title_text='acc', range=[-0.05, 1.05], row=2, col=1)

    if output is not None:
        fig.write_image(output, width=width, height=height, scale=scale)
    else:
        fig.show()
    return fig


def plot_cm(test_outputs, output=None, title='Confusion Matrix', xlab='Predicted Label', ylab='True Label', colorscale='YlOrRd', width=600, height=600, scale=1.0):
    """Plot a confusion matrix from test outputs

    Plot a confusion matrix from test outputs.
    The test outputs are saved in a tab-separated file,
    where the first column is the path to the image, the second column is the true label,
    and the following columns are the predicted probabilities for each class.
    The example of the test outputs is as follows:

    ::

        image  label   leaf     flower   root
        1.JPG  leaf    0.54791  0.20376  0.24833
        2.JPG  root    0.06158  0.02184  0.91658
        3.JPG  leaf    0.70320  0.04808  0.24872
        4.JPG  flower  0.04723  0.90061  0.05216
        5.JPG  flower  0.30027  0.63067  0.06906
        6.JPG  leaf    0.52753  0.43249  0.03998
        7.JPG  root    0.21375  0.14829  0.63796
    

    Args:
        test_outputs (str): A path to a tab-separated file containing test outputs.
        output (str): A file path to save the output images. If not provided, the plot is shown on display.
        width (int): A width of the output image.
        height (int): A height of the output image.
        scale (float): The scale of the output image, which is used to adjust the resolution.
    """

    # data preparation
    test_outputs = pd.read_csv(test_outputs, sep='\t', header=0, comment='#')
    class_labels = test_outputs.columns[2:]
    y_true = test_outputs.iloc[:, 1].values.tolist()
    y_pred = test_outputs.iloc[:, 2:].idxmax(axis=1).values.tolist()
    
    # statistics calculation
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=test_outputs.columns[2:])

    fig = go.Figure(data=go.Heatmap(x=class_labels, y=class_labels, z=cm,
                                    colorscale=colorscale, hoverongaps=False))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab,
                      xaxis=dict(side='bottom'), yaxis=dict(side='left'))
    fig.update_layout(template='ggplot2')


    if output is not None:
        fig.write_image(output, width=width, height=height, scale=scale)
        cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        with open(os.path.splitext(output)[0] + '.txt', 'w') as oufh:
            oufh.write('# Confusion Matrix\n')
            oufh.write('#\tprediction\n')
            cm.to_csv(oufh, sep='\t', header=True, index=True)
    else:
        fig.show()

    return fig



def __generate_source(script_fpath, vanilla=False):
    if not script_fpath.endswith('.py'):
        script_fpath += '.py'

    tmpl = ''
    with open(importlib.resources.files('cvtk').joinpath('tmpl/_torch.py'), 'r') as infh:
        tmpl = infh.readlines()

    if vanilla is True:
        cvtk_modules = [
            {'cvtk.ml.data': [DataLabel]},
            {'cvtk.ml.torchutils': [DataTransform, Dataset, DataLoader, CLSCORE, plot_trainlog, plot_cm]}
        ]
        tmpl = __insert_imports(tmpl, __get_imports(__file__))
        tmpl = __extend_cvtk_imports(tmpl, cvtk_modules)
    
    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__SCRIPTNAME__', os.path.basename(script_fpath))
    tmpl = __del_docstring(tmpl)

    with open(script_fpath, 'w') as fh:
        fh.write(tmpl)
