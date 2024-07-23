import os
import re
import pathlib
import glob
import io
import typing
import base64
import json
import PIL
import PIL.Image
import PIL.ImageOps
import numpy as np
ImageSourceTypes = typing.Union[str, pathlib.Path, bytes, PIL.Image.Image, np.ndarray]


class JsonComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.nan):
            return None
        else:
            return super(JsonComplexEncoder, self).default(obj)



class ImageAnnotation():
    """Image Annotation Class

    Args:
        labels: str | list[str]: The label or list of labels.
        bboxes: list[tuple]|None: The list of bounding boxes in the format of (x, y, w, h).
        masks: list[np.ndarray]|None: NumPy array of maks.
        scores: list[float]|None: The list of scores.
    """
            
    def __init__(self, labels, bboxes=None, masks=None, scores=None):

        if isinstance(labels, str):
            labels = [labels]
        
        if bboxes is not None:
            if len(bboxes) != len(labels):
                raise ValueError('The number of labels and bounding boxes should be the same.')
        else:
            bboxes = [None] * len(labels)
        if masks is not None:
            if len(masks) != len(labels):
                raise ValueError('The number of labels and polygons should be the same.')
        else:
            masks = [None] * len(labels)
        if scores is not None:
            if len(scores) != len(labels):
                raise ValueError('The number of labels and scores should be the same.')
        else:
            scores = [None] * len(labels)

        self.__i = 0
        self.__labels = labels
        self.__bboxes = bboxes
        self.__masks = masks
        self.__scores = scores
        self.__areas = self.__calc_areas()


    def __len__(self):
        return len(self.__labels)


    def __getitem__(self, i):
        return {'label': self.__labels[i],
                'bbox': self.__bboxes[i],
                'mask': self.__masks[i],
                'score': self.__scores[i],
                'area': self.__areas[i]}


    def __iter__(self):
        return self


    def __next__(self):
        if self.__i < len(self):
            i = self.__i
            self.__i += 1
            return self[i]
        else:
            self.__i = 0
            raise StopIteration()


    def __calc_areas(self):
        areas = []
        for bbox, mask in zip(self.__bboxes, self.__masks):
            if mask is not None:
                areas.append(np.sum(mask))
            elif bbox is not None:
                areas.append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
            else:
                areas.append(None)
        return areas


    def dump(self, indent=None, ensure_ascii=True):
        ann_dict = [self[i] for i in range(len(self))]
        return json.dumps(ann_dict, cls=JsonComplexEncoder, indent=indent, ensure_ascii=ensure_ascii)
    
    
    @property
    def labels(self):
        return self.__labels


    @property
    def bboxes(self):
        return self.__bboxes


    @property
    def masks(self):
        return self.__masks
    

    @property
    def scores(self):
        return self.__scores


    def label(self, i):
        return self.__labels[i]
    
    
    def bbox(self, i):
        return self.__bboxes[i]
    

    def mask(self, i):
        return self.__masks[i]
    

    def score(self, i):
        return self.__scores[i]
    



class Image():
    def __init__(self, source, annotations: ImageAnnotation=None):
        self.file_path = source
        self.im = PIL.ImageOps.exif_transpose(PIL.Image.open(source))
        self.annotations = annotations
    
    @property
    def size(self):
        return self.im.size
    
    @property
    def width(self):
        return self.im.width
    
    @property
    def height(self):
        return self.im.height





def imread(source: ImageSourceTypes,
           exif_transpose: bool=True,
           req_timeout: int=60) -> PIL.Image.Image:
    """Open image from various sources

    This function opens image from various sources,
    including file, url, bytes, base64, PIL image, and numpy array
    and convert it to the PIL.Image.Image class instance.
    The format of input image is automatically estimated in the function.
    Image will be transposed based on the EXIF orientation tag if `exif_transpose` is set to True.
    Note that, if 'cv2' format is selected, the image will be in BGR format, compatible with OpenCV.
    
    Args:
        source: str | pathlib.Path | bytes | PIL.Image.Image | np.ndarray: Image source,
            can be a file path, url, bytes, base64, PIL image, or numpy array.
        exif_transpose: bool: Whether to transpose the image based on the EXIF orientation tag.
        req_timeout: int: The timeout for the request to get image from url. Default is 60 seconds.
    
    Returns:
        PIL.Image.Image: Image data.
        
    Examples:
        >>> im = imread('image.jpg')
    """
    im = None

    if isinstance(source, str):
        if re.match(r'https?://', source):
            try:
                import requests
            except ImportError as e:
                raise ImportError('Unable to open image from url. '
                                  'Install requests package to enable this feature.') from e
            try:
                req = requests.get(source, timeout=req_timeout)
                req.raise_for_status()
                return imread(req.content)
            except requests.RequestException as e:
                raise ValueError('Image Not Found.', source) from e
            
        elif source.startswith('data:image'):
            return imread(base64.b64decode(source.split(',')[1]))

        else:
            return imread(pathlib.Path(source))
    
    elif isinstance(source, PIL.Image.Image):
        return source
    
    elif isinstance(source, pathlib.Path):
        im = PIL.Image.open(source)
        if exif_transpose:
            im = PIL.ImageOps.exif_transpose(im)

    elif isinstance(source, (bytes, bytearray)):
        source = np.asarray(bytearray(source), dtype=np.uint8)
        im = PIL.Image.open(io.BytesIO(source))
        if exif_transpose:
            im = PIL.ImageOps.exif_transpose(im)
        
    elif isinstance(source, np.ndarray):
        im = source.copy()
        im = PIL.Image.fromarray(im[..., 2::-1])
    
    else:
        raise ValueError(f'Unable open image file due to unknown type of "{source}".')
    
    if im is None:
        raise ValueError(f'Unable open image file f{source}. Check if the file exists or the url is correct.')

    return im
    



def imconvert(im: ImageSourceTypes,
              format: str='PIL') -> ImageSourceTypes:
    """Convert image format

    Convert image format from any format to the specific format.

    Args:
        im: ImageSourceTypes: Image data in numpy array format.
        format: str: The format of the returned image. Default is 'PIL'.
            Options are 'cv2', 'bytes', 'base64', 'PIL'.
        b_format: str: The image extension to save the image in bytes format. Default is '.jpg'.
    
    Returns:
        ImageSourceTypes: Image data in the specified format.
        
    Examples:
        >>> im = imread('image.jpg')
        >>> imconvert(im, 'cv2')
    """
    def __pil2bytes(im) -> bytes:
        im_buff = io.BytesIO()
        im.save(im_buff, format='JPEG')
        return im_buff.getvalue()

    im = imread(im)

    if format.lower() in ['array', 'cv2', 'cv']:
        return np.array(im)[..., 2::-1]
    elif format.lower() == 'pil':
        return im
    elif format.lower() == 'bytes':
        return __pil2bytes(im)
    elif format.lower() == 'base64':
        return 'data:image/jpeg;base64, ' + \
            base64.b64encode(__pil2bytes(im)).decode('utf-8') 
    elif format.lower() in ['gray', 'grey']:
        return im.convert('L')
    else:
        raise ValueError(f'Unsupported image format "{format}".')




def imresize(im: ImageSourceTypes,
             shape: list[int, int]|tuple[int, int]=None,
             scale: float=None,
             shortest: int=None,
             longest: int=None,
             resample: object=PIL.Image.BILINEAR) -> PIL.Image.Image:
    """Resize the image

    Resize the image to the given shape, scale, shortest, or longest side.

    Args:
        im: ImageSourceTypes: Image data in numpy array format.
        shape: tuple: The shape of the resized image (height, width).
        scale: float: The scale factor to resize the image.
        shortest: int: The shortest side of the image.
        longest: int: The longest side of the image.
        resample: int: The resampling filter. Default is PIL.Image.BILINEAR.
    
    """
    im = imread(im)
    
    if shape is not None:
        im = im.resize(shape, resample=resample)
    elif scale is not None:
        im = im.resize((int(im.width * scale), int(im.height * scale)), resample=resample)
    elif shortest is not None:
        ratio = shortest / min(im.size)
        im = im.resize((int(im.width * ratio), int(im.height * ratio)), resample=resample)
    elif longest is not None:
        ratio = longest / max(im.size)
        im = im.resize((int(im.width * ratio), int(im.height * ratio)), resample=resample)
    else:
        raise ValueError('Specify the shape, scale, shortest, or longest side to resize the image.')
    return im
    


def imwrite(im: ImageSourceTypes, output: str, quality: int=95) -> None:
    """Save image to file

    Args:
        im: ImageSourceTypes: Image data in numpy array format.

    Examples:
        >>> imsave(imread('image.jpg'), 'image.jpg')
        >>> imsave(imread('image.jpg'), 'image.jpg', 100)
    """
    im = imread(im)
    im.save(output, quality=quality)



def imshow(im: ImageSourceTypes|list[ImageSourceTypes], ncol: int|None=None, nrow: int|None=None):
    """Display image using matplotlib.pyplot

    Args:
        im: ImageSourceTypes: Image or list of images to display.
        ncol: int: Number of columns to display the images. Default is None (automatically set).
        nrow: int: Number of rows to display the images. Default is None (automatically set).
    """
    try:
        import math
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('Unable to display image. '
                          'Install matplotlib package to enable image visualization feature.') from e

    if not isinstance(im, (list, tuple)):
        im = [im]

    # set subplot panels
    if ncol is None and nrow is None:
        ncol = nrow = 1
        if len(im) > 1:
            ncol = math.ceil(math.sqrt(len(im)))
            nrow = math.ceil(len(im) / ncol)
    elif ncol is None:
        ncol = math.ceil(len(im) / nrow)
    elif nrow is None:
        nrow = math.ceil(len(im) / ncol)
    
    plt.figure()

    for i_, im_ in enumerate(im):
        plt.subplot(nrow, ncol, i_ + 1)
        plt.imshow(imread(im_))
        if isinstance(im_, str):
            plt.title(os.path.basename(im_))

    plt.show()
    return plt



def imlist(source: str|list[str],
           ext: str|list[str]=['.jpg', '.jpeg', '.png', '.tiff'],
           ignore_case: bool=True) -> list[str]:
    """List all image files from the given sources

    The function recevies image sources as a file path, directory path, or a list of file and directory paths.
    If the source is a directory, the function will recursively search for image files with the given extensions.

    Args:
        source: str | list[str]: The directory path.
        ext: list[str]: The list of file extensions to search for. Default is ['.jpg', '.png', '.tiff'].
        ignore_case: bool: Whether to ignore the case of the file extension. Default is True.

    Returns:
        list: List of image files in the directory.
    """
    im_list = []
    if isinstance(source, str):
        sources = [source]
    if ignore_case:
        ext = [e.lower() for e in ext]

    for source in sources:
        if os.path.isdir(source):
            for f in glob.glob(os.path.join(source, '**', '*'), recursive=True):
                f_ext = os.path.splitext(f)[1]
                if ignore_case:
                    f_ext = f_ext.lower()
                if f_ext in ext:
                    im_list.append(f)
        elif os.path.isfile(source):
            im_list.append(source)
        else:
            raise ValueError(f'The given "{source}" is neither a file nor a directory.')
                
    return im_list








