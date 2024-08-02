import os
import json
import PIL
import PIL.Image
import PIL.ImageFile
import numpy as np
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLabel():
    """Class to treat class labels

    This class is designed to manage class (category) labels for machine learning tasks.
    The class loads class labels from a list, tuple, or text file when creating an instance.
    Methods implemented in the class provide a way to get the class index from the class name and vice versa.

    Args:
        labels (tuple|list|str): A tuple or list,
            or a path to a text file or coco format containing class labels.
            Text file should contain one class name per line.
    
    Examples:
        >>> from cvtk.ml import DataLabel
        >>> 
        >>> labels = ['leaf', 'flower', 'root']
        >>> DataLabel = DataLabel(labels)
        >>> print(DataLabel[1])
        'flower'
        >>> print(DataLabel['flower'])
        1
        >>> len(DataLabel)
        3
        >>> DataLabel.classes
        ['leaf', 'flower', 'root']
        >>> 
        >>> 
        >>> labels = 'labels.txt'
        >>> DataLabel = DataLabel(labels)
        >>> print(DataLabel[1])
        'flower'
        >>> print(DataLabel['flower'])
        1
    """
    def __init__(self, labels):
        if isinstance(labels, list) or isinstance(labels, tuple):
            self.__labels = labels
        elif isinstance(labels, str):
            self.__labels = self.__load_labels(labels)
        else:
            raise TypeError('Expect list, tuple, or str for `labels` but {} was given.'.format(type(labels)))

    def __len__(self):
        return len(self.__labels)


    def __getitem__(self, i):
        if isinstance(i, int) or isinstance(i, str):
            return self.__getitem(i)
        elif isinstance(i, list) or isinstance(i, tuple):
            return [self.__getitem(_) for _ in i]
        else:
            raise TypeError('Expect int or str for `i` to get the class index or name but {} was given.'.format(type(i)))


    def __getitem(self, i):
        if isinstance(i, int):
            return self.__labels[i]
        elif isinstance(i, str):
            return self.__labels.index(i)


    def __load_labels(self, fpath):
        cl = []
        if os.path.splitext(fpath)[1] == '.json':
            with open(fpath, 'r') as fh:
                coco_dict = json.load(fh)
            for cat in sorted(coco_dict['categories'], key=lambda x: x['id']):
                cl.append(cat['name'])
        else:
            with open(fpath, 'r') as fh:
                for _ in fh:
                    _ = _.strip()
                    if _ != '':
                        cl.append(_)
        return cl
    
    
    @property
    def labels(self):
        return self.__labels




class SquareResize():
    """Resize an image to a square shape

    SquareResize provides a function to resize an image to a square.
    The resizing process resizes the length of the long side of the input image to the specified size,
    then adds padding to both sides of the short side of the image to convert the image to a square.

    The background of the padding area is set by default
    to stretch the pixels at both ends of the image as is and then blur them. 
    By specifying `bg_color`, the background of the padding area can be set to a single color.

    Args:
        shape (int): The resolution of the square image.
        bg_color (tuple): The color of the padding area. Default is None.
            If None, the color is extended from both ends of the image.
        resample (int): The resampling filter to use. Default is PIL.Image.BILINEAR.

    Returns:
        The squre image in PIL.Image.Image class.
    
    Examples:
        >>> from cvtk.ml import SquareResize
        >>> 
        >>> squareresize = SquareResize(shape=600)
        >>> img = squareresize('image.jpg')
        >>> img.save('image_square.jpg')
        >>>
        >>> squareresize = SquareResize(shape=600, bg_color=(0, 0, 0))
        >>> img = squareresize('image.jpg')
        >>> img.save('image_square.jpg')
        >>> 
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
    def __init__(self, shape=600, bg_color = None, resample=PIL.Image.BILINEAR):
        self.shape = shape
        self.bg_color = bg_color
        self.resample = resample

    def __call__(self, image, output_fpath=None):
        if isinstance(image, str):
            im = PIL.Image.open(image)
        elif isinstance(image, PIL.Image.Image):
            im = image
        elif isinstance(image, np.ndarray):
            im = PIL.Image.fromarray(image)
        else:
            raise TypeError('Expect str, PIL.Image.Image, or np.ndarray for `image` but {} was given.'.format(type(image)))

        scale_ratio = self.shape / max(im.size)
        im.resize((int(im.width * scale_ratio), int(im.height * scale_ratio)), resample=self.resample)

        w, h = im.size

        im_square = None
        if w == h:
            im_square = im
        else:
            im_array = np.array(im)
            im_square_ = np.zeros([max(w, h), max(w, h), 3])
            if self.bg_color is not None:
                im_square_[:, :, :] = self.bg_color

            if w > h:
                im_square_[0:int(w / 2), :, :] = im_array[0, :, :]
                im_square_[int(w / 2):w, :, :] = im_array[-1, :, :]
                im_square = PIL.Image.fromarray(np.uint8(im_square_))
                im_square = im_square.filter(PIL.ImageFilter.GaussianBlur(3))
                im_square.paste(im, (0, (w - h) // 2))
            else:
                im_square_[0:int(h / 2), :, :] = im_array[:, 0, :]
                im_square_[int(h / 2):h, :, :] = im_array[:, -1, :]
                im_square_ = np.transpose(im_square_, (1, 0, 2))
                im_square = PIL.Image.fromarray(np.uint8(im_square_))
                im_square = im_square.filter(PIL.ImageFilter.GaussianBlur(3))
                im_square.paste(image, ((h - w) // 2, 0))
        
        im_square = im_square.resize((self.shape, self.shape))

        if output_fpath is not None:
            im_square.save(output_fpath)
        
        return im_square
