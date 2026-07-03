import os
import json
import PIL
import PIL.Image
import PIL.ImageFile
import PIL.ImageFilter
import numpy as np
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLabel():
    """Manage class labels for machine learning tasks.

    Loads and manages class (category) labels from various sources (list, tuple, text file, or COCO JSON).
    Provides bidirectional lookup: get label name by index or get index by label name.

    Args:
        labels (list|tuple|str): Class labels. Can be:
            - list or tuple: Direct list of label names
            - str: File path to text file (one label per line) or COCO JSON file

    Attributes:
        labels (list): List of all class labels.

    Raises:
        TypeError: If labels is not list, tuple, or str.
        FileNotFoundError: If file path provided doesn't exist.

    Examples:
        >>> from cvtk.ml.data import DataLabel
        >>> datalabel = DataLabel(['leaf', 'flower', 'root'])
        >>> datalabel[0]
        'leaf'
        >>> datalabel['flower']
        1
        >>> len(datalabel)
        3
        >>> datalabel.labels
        ['leaf', 'flower', 'root']
        >>> datalabel = DataLabel('labels.txt')
    """
    def __init__(self, labels: list|tuple|str):
        """Initialize DataLabel with labels from various sources.

        Args:
            labels (list|tuple|str): Class labels:
                - list/tuple: Direct list of label names
                - str ending in '.json': COCO format JSON file (categories extracted)
                - str (other): Text file with one label per line

        Raises:
            TypeError: If labels is not list, tuple, or str.
            FileNotFoundError: If provided file path doesn't exist.
        """
        if isinstance(labels, list) or isinstance(labels, tuple):
            self.__labels = labels
        elif isinstance(labels, str):
            self.__labels = self.__load_labels(labels)
        else:
            raise TypeError('Expect list, tuple, or str for `labels` but {} was given.'.format(type(labels)))

    def __len__(self):
        """Return the number of classes."""
        return len(self.__labels)


    def __getitem__(self, i):
        """Get label(s) by index or index by label name.

        Args:
            i (int|str|list|tuple): Index or label name(s):
                - int: Get label at this index
                - str: Get index of this label name
                - list/tuple: Get multiple labels/indices (returns list)

        Returns:
            str|int|list: Label name(s) or index(es).

        Raises:
            TypeError: If i is not int, str, list, or tuple.
            IndexError: If int index out of range.
            ValueError: If str label not found.
        """
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
        """List of all class labels.
        
        Returns:
            list: All label names in order.
        """
        return self.__labels


    def save(self, output):
        """Save class labels to a text file.

        Saves one label per line in plain text format.

        Args:
            output (str): File path for output text file.

        Returns:
            None. File saved to disk.

        Examples:
            >>> datalabel = DataLabel(['cat', 'dog', 'bird'])
            >>> datalabel.save('labels.txt')
        """
        with open(output, 'w') as fh:
            fh.write('\n'.join(self.__labels))    
    


class SquareResize():
    """Resize image to square with padding and optional color background.

    Resizes an image to a square by:
    1. Scaling the longest side to target shape size
    2. Padding the shorter side with either blurred edge pixels or solid color

    Useful as a preprocessing transform for image classification tasks.

    Args:
        shape (int): Target square size (width and height in pixels). Default is 600.
        bg_color (tuple[int,int,int]|None): RGB color for padding area. If None, uses blurred edge pixels. Default is None.
        resample (int): PIL resampling filter for scaling. Default is PIL.Image.BILINEAR.

    Returns:
        PIL.Image.Image: Square image of shape (shape, shape) in RGB mode.

    Examples:
        >>> from cvtk.ml.data import SquareResize
        >>> squareresize = SquareResize(shape=600)
        >>> img = squareresize('image.jpg')
        >>> img.save('image_square.jpg')
        >>> squareresize = SquareResize(shape=600, bg_color=(0, 0, 0))
        >>> img = squareresize('image.jpg')
        >>> img.save('image_square.jpg')
        >>> import torchvision.transforms
        >>> transform = torchvision.transforms.Compose([
        ...     SquareResize(256),
        ...     torchvision.transforms.RandomHorizontalFlip(0.5),
        ...     torchvision.transforms.RandomAffine(45),
        ...     torchvision.transforms.ToTensor(),
        ...     torchvision.transforms.Normalize([0.485, 0.456, 0.406],
        ...                                      [0.229, 0.224, 0.225])
        ... ])
    """
    def __init__(self, shape: int=600, bg_color: tuple[int, int, int]|None=None, resample: object=PIL.Image.BILINEAR):
        """Initialize SquareResize transform.

        Args:
            shape (int): Target square size in pixels. Default is 600.
            bg_color (tuple[int,int,int]|None): RGB color for padding (0-255). If None, uses blurred edge pixels. Default is None.
            resample (int): PIL resampling filter (e.g., PIL.Image.BILINEAR, PIL.Image.LANCZOS). Default is PIL.Image.BILINEAR.
        """
        self.shape = shape
        self.bg_color = bg_color
        self.resample = resample

    def __call__(self, image, output_fpath=None):
        """Apply square resize to image.

        Resizes image to square by scaling longest side and padding shorter side.

        Args:
            image (str|PIL.Image.Image|np.ndarray): Image to resize:
                - str: File path to image
                - PIL.Image.Image: PIL image object
                - np.ndarray: Numpy array (converted to PIL image)
            output_fpath (str|None): If provided, saves result to this file path. Default is None.

        Returns:
            PIL.Image.Image: Square image of shape (shape, shape) in RGB.

        Raises:
            TypeError: If image is not str, PIL.Image.Image, or np.ndarray.

        Examples:
            >>> squareresize = SquareResize(256)
            >>> result = squareresize('photo.jpg')
            >>> result = squareresize('photo.jpg', output_fpath='square.jpg')
        """
        if isinstance(image, str):
            im = PIL.Image.open(image)
        elif isinstance(image, PIL.Image.Image):
            im = image
        elif isinstance(image, np.ndarray):
            im = PIL.Image.fromarray(image)
        else:
            raise TypeError('Expect str, PIL.Image.Image, or np.ndarray for `image` but {} was given.'.format(type(image)))

        scale_ratio = self.shape / max(im.size)
        im = im.resize((int(im.width * scale_ratio), int(im.height * scale_ratio)), resample=self.resample)

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
                im_square.paste(im, ((h - w) // 2, 0))
        
        im_square = im_square.resize((self.shape, self.shape))

        if output_fpath is not None:
            im_square.save(output_fpath)
        
        return im_square
