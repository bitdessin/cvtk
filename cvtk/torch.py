import os
import random
import gzip
import numpy as np
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import PIL.ImageFilter
try:
    import torch
except ImportError as e:
    raise ImportError('Unable to import torch. '
                      'Install torch package to enable this feature.') from e

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True



class DataClass():
    """A class to treat class labels

    DataClass is designed to load class labels from a list, tuple or text file.
    It provides a way to get class index from class name, and vice versa.

    Args:
        class_labels (tuple|list|str): A tuple or list,
            or a path to a text file containing class labels.
            
    
    Examples:
        >>> from cvtk import DataClass
        >>> class_labels = ['leaf', 'flower', 'root']
        >>> dataclass = DataClass(class_labels)
        >>> print(dataclass[1])
        'flower'
        >>> print(dataclass['flower'])
        1
        >>> len(dataclass)
        3
        >>> dataclass.classes
        ['leaf', 'flower', 'root']
        >>> 
        >>> 
        >>> class_labels = 'class_labels.txt'
        >>> dataclass = DataClass(class_labels)
        >>> print(dataclass[1])
        'flower'
        >>> print(dataclass['flower'])
        1
    """
    def __init__(self, class_labels):
        if isinstance(class_labels, list) or isinstance(class_labels, tuple):
            self.classes = class_labels
        elif isinstance(class_labels, str):
            self.classes = self.__load_classnames(class_labels)
        else:
            raise TypeError('Expect list, tuple, or str for `class_labels` but {} was given.'.format(type(class_labels)))


    def __len__(self):
        return len(self.classes)


    def __getitem__(self, i):
        if isinstance(i, int) or isinstance(i, str):
            return self.__getitem(i)
        elif isinstance(i, list) or isinstance(i, tuple):
            return [self.__getitem(_) for _ in i]
        else:
            raise TypeError('Expect int or str for `i` to get the class index or name but {} was given.'.format(type(i)))


    def __getitem(self, i):
        if isinstance(i, int):
            return self.classes[i]
        elif isinstance(i, str):
            return self.classes.index(i)


    def __load_classnames(self, fpath):
        cl = []
        with open(fpath, 'r') as f:
            cl_ = f.read().splitlines()
        for cl__ in cl_:
            if (cl__ != ''):
                cl.append(cl__)
        return cl





class SquareResize:
    """Resize an image to a square shape

    SquareResize provides a function to resize an image to a square shape.
    The short edge of the image is changed to the same length as the long edge by adding padding,
    and then the image is resized to a specific resolution.
    The background of the padding area is set as extended from both ends of the image by default,
    but it can be changed by the user with `bg_color`.
    This class can be used in a pipeline of image processing with `torchvision.transforms.Compose`.

    Args:
        shape (int): The resolution of the square image.
        bg_color (tuple): The color of the padding area. Default is None.
            If None, the color is extended from both ends of the image.
    
    Examples:
        >>> from cvtk import SquareResize
        >>> sr = SquareResize(shape=600)
        >>> img = sr('image.jpg')
        >>> img.save('image_square.jpg')
        >>>
        >>> sr = SquareResize(shape=600, bg_color=(255, 255, 255))
        >>> img = sr('image.jpg')
        >>> img.save('image_square.jpg')
        >>>
        >>> import torchvision.transforms
        >>> transform = torchvision.transforms.Compose([
                SquareResize(256),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomAffine(45),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
    """
    def __init__(self, shape=600, bg_color = None):
        self.shape = shape
        self.bg_color = bg_color

    def __call__(self, image, output_fpath=None):
        if isinstance(image, PIL.Image.Image):
            pass
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(np.uint8(image))
        elif isinstance(image, str):
            image = PIL.Image.open(image)
        else:
            raise TypeError('Expect PIL.Image.Image, np.ndarray, or str for `image` but {} was given.'.format(type(img)))

        w, h = image.size

        image_square = None
        if w == h:
            image_square = image
        else:
            image_array = np.array(image)
            image_square_ = np.zeros([max(w, h), max(w, h), 3])
            if self.bg_color is not None:
                image_square_[:, :, :] = self.bg_color

            if w > h:
                image_square_[0:int(w / 2), :, :] = image_array[0, :, :]
                image_square_[int(w / 2):w, :, :] = image_array[-1, :, :]
                image_square = PIL.Image.fromarray(np.uint8(image_square_))
                image_square = image_square.filter(PIL.ImageFilter.GaussianBlur(3))
                image_square.paste(image, (0, (w - h) // 2))
            else:
                image_square_[0:int(h / 2), :, :] = image_array[:, 0, :]
                image_square_[int(h / 2):h, :, :] = image_array[:, -1, :]
                image_square_ = np.transpose(image_square_, (1, 0, 2))
                image_square = PIL.Image.fromarray(np.uint8(image_square_))
                image_square = image_square.filter(PIL.ImageFilter.GaussianBlur(3))
                image_square.paste(image, ((h - w) // 2, 0))
        
        image_square = image_square.resize((self.shape, self.shape))

        if output_fpath is not None:
            image_square.save(output_fpath)
        
        return image_square
    


class DatasetLoader(torch.utils.data.Dataset):
    """A class to load images for training or testing

    DatasetLoader is designed to load images for training or testing with PyTorch.
    The class can load images from a directory, a list, a tuple, or a tab-separated file.

    Args:
        dataset (str|list|tuple): A path to a directory, a list, a tuple, or a tab-separated file.
            If a path to a directory is given, the class loads all images in the directory.
            If a list or a tuple is given, the class loads images from the list or tuple.
            If a tab-separated file is given, the class loads images from the file.
        dataclass (DataClass): A class to treat class labels.
        transform (None|torchvision.transforms.Compose): A pipeline of image processing.
        balance_train (bool): If True, the number of images in each class is balanced

    Examples:
        >>> from cvtk import DataClass, DatasetLoader
        >>> class_labels = ['leaf', 'flower', 'root']
        >>> dataclass = DataClass
        >>> dataset = 'dataset.txt'
        >>> transform = torchvision.transforms.Compose([
                SquareResize(256),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomAffine(45),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
        >>> dataset = DatasetLoader(dataset, dataclass, transform, balance_train=True)
        >>> print(len(dataset))
        100
        >>> img, label = dataset[0]
        >>> print(img.shape)
        >>> print(label)
    """

    def __init__(self,
                 dataset,
                 dataclass,
                 transform=None,
                 balance_train=False):
        
        self.transform = transform
        self.balance_train = balance_train
        self.x , self.y = self.__load_images(dataset, dataclass)

    def __load_images(self, dataset, dataclass):
        x = []
        y = []
        if isinstance(dataset, str):
            if os.path.isfile(dataset):
                # load a single image, or images from a tab-separated file
                if os.path.splitext(dataset)[1].lower() in ['.jpg', '.jpeg', '.png']:
                    # load a single image file
                    x = [dataset]
                    y = [None]
                else:
                    # load a tab-separated file
                    if dataset.endswith('.gz') or dataset.endswith('.gzip'):
                        trainfh = gzip.open(dataset, 'rt')
                    else:
                        trainfh = open(dataset, 'r')
                    trainfh.readline()
                    x = []
                    y = []
                    for line in trainfh:
                        words = line.rstrip().split('\t')
                        x.append(words[0])
                        # set label to None if the file does not contain the label column in the second column
                        if len(words) >= 2:
                            y.append(dataclass[words[1]])
                        else:
                            y.append(None)
                    trainfh.close()
            elif os.path.isdir(dataset):
                # load images from a directory without labels
                for root, dirs, files in os.walk(dataset):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']:
                            x.append(os.path.join(root, f))
                            y.append(None)
        elif isinstance(dataset, list) or isinstance(dataset, tuple):
            # load images from a list or tuple
            for d in dataset:
                if isinstance(d, list) or isinstance(d, tuple):
                    if len(d) >= 2:
                        x.append(d[0])
                        y.append(dataclass[d[1]])
                    else:
                        x.append(d[0])
                        y.append(None)
                else:
                    x.append(d)
                    y.append(None)

        if self.balance_train:
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
        # upsample the number of images in minority classes
        # to the number of images in majority class
        y0_idx = []
        y1_idx = []
        for i in range(len(y)):
            if y[i] == 0:
                y0_idx.append(i)
            elif y[i] == 1:
                y1_idx.append(i)

        if len(y0_idx) > len(y1_idx):
            y1_idx_sampled = random.choices(y1_idx, k=len(y0_idx) - len(y1_idx))
            y.extend([y[i] for i in y1_idx_sampled])
            x.extend([x[i] for i in y1_idx_sampled])
        elif len(y0_idx) < len(y1_idx):
            y0_idx_sampled = random.choices(y0_idx, k=len(y1_idx) - len(y0_idx))
            y.extend([y[i] for i in y0_idx_sampled])
            x.extend([x[i] for i in y0_idx_sampled])

        return x, y


class CLSCORE():
    """A class provides training and inference functions for a classification model

    CLSCORE is a class that provides training and inference functions for a classification model.
    It requires a model, a data class, and a temporary directory path to save intermediate checkpoints and training logs automatically.

    Args:
        model (torch.nn.Module): a model to be trained
        dataclass (DataClass): a data class that contains class labels
        temp_dirpath (str): a temporary directory path to save intermediate checkpoints and training logs

    Attributes:
        device (str): a device to run the model
        dataclass (DataClass): a data class that contains class labels
        model (torch.nn.Module): a model to be trained
        temp_dirpath (str): a temporary directory path to save intermediate checkpoints and training logs
        train_stats (dict): a dictionary to save training statistics
        test_stats (dict): a dictionary to save test statistics

    Examples:
        >>> import torch
        >>> import torchvision
        >>> from cvtk import DataClass
        >>>
        >>> # dataset
        >>> dataclass = DataClass('dataclass.txt')
        >>> dataloaders = {
        >>>     'train': train_dataloader,
        >>>     'valid': valid_dataloader,
        >>>     'test': test_dataloader
        >>> }
        >>>
        >>> # model
        >>> model = torchvision.models.efficientnet_b7(
        >>>     weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)
        >>> model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(dataclass.classes))
        >>>
        >>> # train and inference
        >>> clscore = CLSCORE(model, dataclass)
        >>> clscore.train(dataloaders, epochs=20, lr=0.01)
        >>>
        >>> dataloader = test_dataloader
        >>> probs = clscore.inference(dataloader)
    """

    def __init__(self, model, dataclass, temp_dirpath=None):
        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # data class
        if isinstance(dataclass, str):
            dataclass = DataClass(dataclass)
        elif not isinstance(dataclass, DataClass):
            raise TypeError('Invalid type: {}'.format(type(dataclass)))
        self.dataclass = dataclass

        # model
        self.model = model
        self.model = self.model.to(self.device)
        
        # create variables to save intermediate checkpoints and training logs
        self.temp_dirpath = self.__init_tempdir(temp_dirpath)
        self.train_stats = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': []
        }
        self.test_stats = None

    
    def __init_tempdir(self, temp_dirpath):
        if temp_dirpath is None:
            temp_dirpath = os.path.join(
                os.getcwd(),
                '{}_{}'.format(str(uuid.uuid4()).replace('-', '')[0:8],
                              datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        if not os.path.exists(temp_dirpath):
            os.makedirs(temp_dirpath)
        return temp_dirpath


    def train(self, dataloaders, epochs=20,  lr=0.01, resume=False):
        """Train the model with the provided dataloaders

        Train the model with the provided dataloaders. The training statistics are saved in the temporary directory.

        Args:
            dataloaders (dict): a dictionary of dataloaders for training, validation, and test
            epochs (int): the number of epochs to train the model
            lr (float): the learning rate for the optimizer
            resume (bool): resume training from the last checkpoint if True
        
        Examples:
            >>> import torch
            >>> import torchvision
            >>> from cvtk import DataClass
            >>>
            >>> # dataset
            >>> dataclass = DataClass
            >>> dataloaders = {
            >>>     'train': train_dataloader,
            >>>     'valid': valid_dataloader,
            >>>     'test': test_dataloader
            >>> }
            >>>
            >>> # model
            >>> model = torchvision.models.efficientnet_b7(
            >>>     weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)
            >>> model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(dataclass.classes))
            >>>
            >>> # train
            >>> clscore = CLSCORE(model, dataclass)
            >>> clscore.train(dataloaders, epochs=20, lr=0.01)
        """

        # training params
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        # set up train/validation datasets
        if 'train' not in dataloaders:
            raise ValueError('train dataset is not provided')

        # resume training from the last checkpoint if resume is True
        last_epoch = 0
        if resume:
            last_epoch = self.__update_model_weight()

        # train the model
        for epoch in range(last_epoch + 1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}')

            # training and validation
            self.train_stats['epoch'].append(epoch)
            for phase in ['train', 'valid']:
                loss, acc, probs = self.__train(dataloaders[phase], criterion, optimizer, phase)
                self.train_stats[f'{phase}_loss'].append(loss)
                self.train_stats[f'{phase}_acc'].append(acc)
                if loss is not None and acc is not None:
                    print(f'{phase} loss: {loss:.4f}, acc: {acc:.4f}')

            # test the model if dataset is provided at the last epoch
            if epoch == epochs and dataloaders['test'] is not None:
                loss, acc, probs = self.__train(dataloaders['test'], criterion, optimizer, phase)
                self.test_stats = {
                    'dataset': dataloaders['test'].dataset,
                    'loss': loss,
                    'acc': acc,
                    'probs': probs
                }
            
            self.save(os.path.join(self.temp_dirpath, f'checkpoint_latest.pth'))



    def __update_model_weight(self):
        last_epoch = 0

        trainstats_fpath = os.path.join(self.temp_dirpath, 'train_stats.txt')
        chk_fpath = os.path.join(self.temp_dirpath, 'checkpoint_latest.pth')
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



    def __train(self, dataloader, criterion, optimizer, phase):
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
            print(inputs)
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

            #running_loss += loss.item() * inputs.size(0)
            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            probs.append(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

        epoch_loss = running_loss.double().item() / len(dataloader.dataset)
        epoch_acc = running_corrects.double().item() / len(dataloader.dataset)
        probs = np.concatenate(probs, axis=0).tolist()
        return epoch_loss, epoch_acc, probs



    def __str(self, s):
        if s is None:
            return 'NA'
        return str(s)


    def __write_train_stats(self, output_log_fpath):
        with open(output_log_fpath, 'w') as fh:
            fh.write('\t'.join(self.train_stats.keys()) + '\n')
            for vals in zip(*self.train_stats.values()):
                fh.write('\t'.join([self.__str(v) for v in vals]) + '\n')


    def __write_test_outputs(self, output_log_fpath):
        with open(output_log_fpath, 'w') as fh:
            fh.write('# loss: {}\n'.format(self.test_stats['loss']))
            fh.write('# acc: {}\n'.format(self.test_stats['acc']))
            fh.write('\t'.join(['image', 'label'] + self.dataclass.classes) + '\n')
            for x_, y_, p_ in zip(self.test_stats['dataset'].x, self.test_stats['dataset'].y, self.test_stats['probs']):
                fh.write('{}\t{}\t{}\n'.format(
                    x_,
                    self.dataclass.classes[y_],
                    '\t'.join([str(_) for _ in p_])))

    def save(self, output_fpath):
        """Save the model and training statistics

        Save the model and training statistics in the provided output file path. The output file path should end with '.pth'.

        Args:
            output_fpath (str): an output file path to save the model and training statistics
        """
        if not output_fpath.endswith('.pth'):
            output_fpath += '.pth'
        self.model = self.model.to('cpu')
        torch.save(self.model.state_dict(), output_fpath)
        self.model = self.model.to(self.device)

        output_log_fpath = os.path.splitext(output_fpath)[0] + '.train_stats.txt'
        self.__write_train_stats(output_log_fpath)

        if self.test_stats is not None:
            output_log_fpath = os.path.splitext(output_fpath)[0] + '.test_outputs.txt'
            self.__write_test_outputs(output_log_fpath)


    def inference(self, dataloader):
        """Inference the model with the provided dataloader

        Inference the model with the provided dataloader. The output is a list of probabilities for each class.

        Args:
            dataloader (torch.utils.data.DataLoader): a dataloader for inference
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        probs = []
        for inputs in dataloader:
            inputs = inputs[0].to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
            probs.append(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

        probs = np.concatenate(probs, axis=0).tolist()
        return probs


