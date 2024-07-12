import numpy as np
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import PIL.ImageFilter
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import cvtk


class DataClass():
    """Class to treat class labels

    This class is designed to manage class (category) labels for machine learning tasks.
    The class loads class labels from a list, tuple, or text file when creating an instance.
    Methods implemented in the class provide a way to get the class index from the class name and vice versa.

    Args:
        class_labels (tuple|list|str): A tuple or list,
            or a path to a text file containing class labels.
            Text file should contain one class name per line.
    
    Examples:
        >>> from cvtk.ml import DataClass
        >>> 
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
        im = cvtk.imread(image, format='PIL')

        scale_ratio = self.shape / max(im.size)
        im.resize((int(im.width * scale_ratio), int(im.height * scale_ratio)), resample=self.resample)

        w, h = im.size

        image_square = None
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
    
