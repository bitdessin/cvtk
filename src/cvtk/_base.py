import os
import re
import random
import copy
import pathlib
import glob
import io
import typing
import base64
import json
import PIL
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
ImageSourceTypes = typing.Union[str, pathlib.Path, bytes, PIL.Image.Image, np.ndarray]


class JsonComplexEncoder(json.JSONEncoder):
    """Convert objects to JSON serializable format

    Examples:
        >>> obj = {'a': np.array([0, 1, 2]), 'b': [0, 1, 2], 'c': 'hello wolrd'}
        >>> json.dumps(obj, cls=JsonComplexEncoder)
    """
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
            return super().default(obj)


class Annotation():
    """A class to store image annotations including bounding boxes and masks

    The class store image annotations of the coordinates of bounding boxes,
    the binary masks, and the labels and scores corresponding to the bounding boxes or masks.
    The areas of bounding boxes or masks are automatically calculated from bouding boxes or masks and stored in this class.

    Args:
        labels: The labels for the bounding boxes or masks.
        bboxes: The list of bounding boxes in the format of (x1, y1, x2, y2).
        masks: NumPy array or 2-d list of maks.
        scores: The list of scores.
    
    Attributes:
        labels: The labels for the bounding boxes or masks.
        bboxes: The list of bounding boxes in the format of (x1, y1, x2, y2).
        masks: NumPy array or 2-d list of maks.
        scores: The list of scores.
        areas: The list of areas of bounding boxes or masks.

    Examples:
        >>> labels = ['leaf', 'flower', 'root']
        >>> bboxes = [[0, 0, 10, 10],
        ...           [10, 10, 20, 20],
        ...           [20, 20, 30, 30]]
        >>> masks = [np.random.randint(2, (240, 321)).tolist(),
        ...          np.random.randint(2, (240, 321)).tolist(),
        ...          np.random.randint(2, (240, 321)).tolist()]
        >>> scores = [0.9, 0.8, 0.7]
        >>>
        >>> ann = Annotation(labels, bboxes, masks, scores)
        >>>
        >>> ann[0]
        {'label': 'leaf', 'bbox': (0, 0, 10, 10), 'mask': array([[0, 1, 0, ..., 0, 1, 0],
        ...                                                        [1, 0, 0, ..., 1, 0, 0],
        ...                                                        ...,
        ...                                                        [0, 1, 0, ..., 0, 1, 0],
        ...                                                        [1, 0, 0, ..., 1, 0, 0]]),
        ... 'score': 0.9, 'area': 100}
        >>>
        >>> ann.labels
        ['leaf', 'flower', 'root']
        >>>
        >>> ann.dump()
        '[{"label": "leaf", "bbox": [0, 0, 10, 10], "mask": [[0, 1, 0, ..., 0, 1, 0], [1, 0, 0, ..., 1, 0, 0], ..., [0, 1, 0, ..., 0, 1, 0], [1, 0, 0, ..., 1, 0, 0]], "score": 0.9, "area": 100}, {"label": "flower", "bbox": [10, 10, 20, 20], "mask": [[0, 1, 0, ..., 0, 1, 0], [1, 0, 0, ..., 1, 0, 0], ..., [0, 1, 0, ..., 0, 1, 0], [1, 0, 0, ..., 1, 0, 0]], "score": 0.8, "area": 100}, {"label": "root", "bbox": [20, 20, 30, 30], "mask": [[0, 1, 0, ..., 0, 1, 0], [1, 0, 0, ..., 1, 0, 0], ..., [0, 1, 0, ..., 0, 1, 0], [1, 0, 0, ..., 1, 0, 0]], "score": 0.7, "area": 100}]'
    """
            
    def __init__(self,
                 labels:list[str],
                 bboxes: list[list]|list[tuple]|None=None,
                 masks: list[list[int]]|np.ndarray|None=None,
                 scores: list[float]|None=None):

        if isinstance(labels, str):
            labels = [labels]
        
        if bboxes is not None:
            if len(bboxes) != len(labels):
                raise ValueError('The number of labels and bounding boxes should be the same.')
        else:
            bboxes = [None] * len(labels)
        if masks is not None:
            if len(masks) != len(labels):
                raise ValueError('The number of masks and polygons should be the same.')
        else:
            masks = [None] * len(labels)
        if scores is not None:
            if len(scores) != len(labels):
                raise ValueError('The number of labels and scores should be the same.')
        else:
            scores = [None] * len(labels)

        self.__i = 0
        self.__labels = labels
        self.__bboxes = self.__as_tuple(bboxes)
        self.__masks = self.__as_array(masks)
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
        

    def __as_tuple(self, bboxes):
        if bboxes is not None:
            bboxes = [tuple(bbox) if bbox is not None else None for bbox in bboxes]
        return bboxes


    def __as_array(self, masks):
        if masks is not None:
            masks = [np.array(mask) if mask is not None else None for mask in masks]
        return masks


    def __calc_areas(self):
        areas = []
        for bbox, mask in zip(self.__bboxes, self.__masks):
            if mask is not None:
                areas.append(int(np.sum(mask)))
            elif bbox is not None:
                areas.append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
            else:
                areas.append(None)
        return areas
    
    
    @property
    def labels(self) -> list[str]:
        return self.__labels


    @property
    def bboxes(self) -> list[tuple]:
        return self.__bboxes


    @property
    def masks(self) -> list[np.ndarray]:
        return self.__masks
    

    @property
    def scores(self) -> list[float]:
        return self.__scores
    
    
    def dump(self, indent: int|None=None, ensure_ascii: bool=True) -> str:
        """Dump the annotation data to string in JSON format.

        Args:
            indent (int): The indentation of the JSON string. Default is `None`.
            ensure_ascii (bool): Ensure the string is ASCII. Default is `True`.

        Returns:
            str: JSON string of the annotation data.

        Examples:
            >>> ann = ImageAnnotation(['leaf', 'flower', 'root'], [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
            >>> ann.dump()
            '[{"label": "leaf", "bbox": [0, 0, 10, 10], "mask": null, "score": null, "area": 100}, {"label": "flower", "bbox": [10, 10, 20, 20], "mask": null, "score": null, "area": 100}, {"label": "root", "bbox": [20, 20, 30, 30], "mask": null, "score": null, "area": 100}]'
        """
        ann_dict = [self[i] for i in range(len(self))]
        return json.dumps(ann_dict, cls=JsonComplexEncoder, indent=indent, ensure_ascii=ensure_ascii)



class Image():
    """A class to store image data and annotations
    
    The class store image data and annotations including bounding boxes and masks.

    Args:
        source: The path to the image file.
        annotations: The annotations for the image.
    
    Attributes:
        file_path: The path to the image file.
        annotations: The annotations for the image.
        size: The size of the image (width, height).
        width: The width of the image.
        height: The height of the image.
    
    Examples:
        >>> im = Image('image.jpg')
        >>> im.size
        (321, 240)
        >>> im.width
        321
        >>> im.height
        240
        >>>
        >>> labels = ['leaf', 'flower', 'root']
        >>> bboxes = [[0, 0, 10, 10],
        ...           [10, 10, 20, 20],
        ...           [20, 20, 30, 30]]
        >>> ann = ImageAnnotation(labels, bboxes)
        >>> im = Image('image.jpg', ann)
        >>> im.annotations
        <cvtk.base.ImageAnnotation object at 0x7f9d5f4b0f10>
        >>>
    """
    def __init__(self, source, annotations: Annotation|None=None):
        im = imread(source)
        self.__source = source
        self.__size = im.size
        self.__width = im.width
        self.__height = im.height
        self.__annotations = annotations


    @property
    def source(self):
        return self.__source


    @property
    def size(self):
        return self.__size
    

    @property
    def width(self):
        return self.__width
    
    
    @property
    def height(self):
        return self.__height


    @property
    def annotations(self):
        return self.__annotations


    def draw(self,
             format: str='bbox',
             output: str|None=None,
             cutoff: float=0.5,
             label: bool=True,
             score: bool=True,
             font: PIL.ImageFont.ImageFont|None=None,
             col: dict|None=None) -> PIL.Image.Image:
        """Plot an image with annotations
        
        Plot an image with annotations including bounding boxes and masks.

        Args:
            format: The format of the annotations to plot. Default is 'bbox'.
                Options are 'bbox', 'segm', 'mask'.
            output: The path to save the plotted image. Default is None.
            cutoff: The cutoff score to plot the annotations. Default is 0.5.
            label: Whether to plot the labels. Default is True.
            score: Whether to plot the scores. Default is True.
            cols: The color dictionary for the annotations. Default is None.
        """
        import skimage.measure

        im = imread(self.__source)
        imdraw = PIL.ImageDraw.Draw(im)

        if font is None:
            font = PIL.ImageFont.load_default(max([10, int(im.height / 50), int(im.width / 50)]))
        if col is None:
            col = {'___UNDEF___': (random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255))}
        outline_width = max([5, int(im.height / 200), int(im.width / 200)])

        for ann in self.__annotations:
            if ('score' in ann) and (ann['score'] < cutoff):
                continue

            cl = '___UNDEF___'
            if 'label' in ann:
                cl = ann['label']
                if cl not in col:
                    col[cl] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            if ('bbox' in format) and ('bbox' in ann) and (ann['bbox'] is not None):
                x1, y1, x2, y2 = ann['bbox']
                imdraw.rectangle([(x1, y1), (x2, y2)], outline = col[cl], width=outline_width)

            if ('segm' in format) and ('mask' in ann) and (ann['mask'] is not None):
                for contour in skimage.measure.find_contours(ann['mask'], 0.5):
                    imdraw.line([tuple([c[1], c[0]]) for c in contour.tolist()], fill = col[cl], width=outline_width)

            if ('mask' in format) and ('mask' in ann) and (ann['mask'] is not None):
                mask = copy.deepcopy(ann['mask'])
                im = PIL.Image.fromarray((mask * 255).astype(np.uint8))
            
            if ('rgbmask' in format) and ('mask' in ann) and (ann['mask'] is not None):
                mask = np.zeros((im.size[1], im.size[0], 3))
                for i in range(3):
                    mask[:, :, i][ann['mask'] > 0] = col[cl][i]
                im = PIL.Image.fromarray(mask.astype(np.uint8))

            if label and ('label' in ann):
                cl_ = ann['label']
                if score and ('score' in ann):
                    cl_ = f"{ann['label']} ({ann['score']:.2f})"
                x1, y1, x2, y2 = ann['bbox']
                imdraw.text((x1 + int(outline_width * 1.5), y1), cl_, font=font, fill=col[cl])

        if output is not None:
            im.save(output)
        return im



class ImageDeck():
    """A class to store a deck of images
    
    The class store a deck of images and annotations including bounding boxes and masks.

    Args:
        images: The list of images.

    """

    def __init__(self, images: Image|list[Image]):
        self.images = images

        if isinstance(images, Image):
            self.images = [images]
        elif isinstance(images, ImageDeck):
            self.images = copy.deepcopy(images.images)
        elif isinstance(images, (list, tuple)):
            for i, im in enumerate(images):
                if not isinstance(im, Image):
                    raise ValueError(f'Invalid image at index {i}.')
            self.images = images
        else:
            raise ValueError(f'ImageDeck only receives Image or list of Image object.')

        self.__i = 0

    
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

    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, i):
        return self.images[i]
    

    def append(self, image: Image):
        """Append an image to the deck
        
        """
        if not isinstance(image, Image):
            raise ValueError(f'Invalid image.')
        self.images.append(image)


    def extend(self, images: Image|list[Image]):
        """Extend images to the deck
        
        """
        if isinstance(images, Image):
            self.images.append(images)
        elif isinstance(images, (list, tuple)):
            for i, im in enumerate(images):
                if not isinstance(im, Image):
                    raise ValueError(f'Invalid image at index {i}.')
            self.images.extend(images)
        else:
            raise ValueError(f'Invalid image deck.')
    
    
    def format(self, format: str='cvtk', datalabel: str|None=None) -> dict:
        """Format the images in the deck
        
        """

        if format.lower() == 'cvtk':
            deck_dict = self.__format_cvtk()
        elif format.lower() == 'coco':
            deck_dict = self.__format_coco(datalabel)
        #elif format.lower() in ['voc', 'pascal', 'xml']:
        #    deck_dict = self.__format_voc()
        else:
            raise ValueError(f'Invalid format "{format}".')
        
        return deck_dict
    

    def dump(self, output: str, format: str='cvtk', datalabel: str|None=None, indent: int|None=None, ensure_ascii: bool=True):
        """Dump the images in the deck to a file
        
        """

        deck_dict = self.format(format, datalabel)
        with open(output, 'w') as fh:
            json.dump(deck_dict, fh, ensure_ascii=ensure_ascii, indent=indent, cls=JsonComplexEncoder)


    def __format_cvtk(self) -> dict:
        images = []
        for i, im in enumerate(self.images):
            images.append({
                'file_path': im.source,
                'annotations': [ann for ann in im.annotations]
            })
        return {'data': images}


    def __format_coco(self, datalabel) -> dict:
        import pycocotools
        import pycocotools.mask

        images = []
        annotations = []
        categories = []
        cate2id = {}

        cate_id = 0
        if datalabel is not None:
            for label in datalabel:
                cate_id += 1
                cate2id[label] = cate_id
                categories.append({
                    'id': cate_id,
                    'name': label
                })
        else:
            cate_ = set()
            for im in self.images:
                for ann in im.annotations:
                    if 'label' in ann:
                        cate_.add(ann['label'])
            for label in sorted(cate_):
                cate_id += 1
                cate2id[label] = cate_id
                categories.append({
                    'id': cate_id,
                    'name': label
                })

        img_id = 0
        ann_id = 0
        for im in self.images:
            img_id += 1
            images.append({
                'file_name': os.path.basename(im.source),
                'height': im.height,
                'width': im.width,
                'id': img_id
            })
            for ann in im.annotations:
                ann_id += 1
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cate2id[ann['label']],
                    'bbox': self.__xyxy2xywh(ann['bbox']),
                    'score': ann['score'],
                    'area': ann['area'],
                    'iscrowd': 0
                })
                if 'mask' in ann and ann['mask'] is not None:
                    rle_mask = pycocotools.mask.encode(np.asfortranarray(ann['mask']).astype(np.uint8))
                    annotations[-1]['segmentation'] = {
                        'size': rle_mask['size'],
                        'counts': rle_mask['counts'].decode()
                    }
        
        return {'images': images, 'annotations': annotations, 'categories': categories}


    def __xyxy2xywh(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


#    def __format_voc(self) -> dict:
#        pass


def imread(source,
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
        source (str, pathlib.Path, bytes, PIL.Image.Image, np.ndarray): Image source,
            can be a file path, url, bytes, base64, PIL image, or numpy array.
        exif_transpose (bool): Whether to transpose the image based on the EXIF orientation tag.
        req_timeout (int): The timeout for the request to get image from url. Default is 60 seconds.
    
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
        im (ImageSourceTypes): Image source, can be a file path, url, bytes, base64, PIL image, or numpy array.
        format (str): The format of the returned image. Default is 'PIL'.
            Options are 'cv2' (or 'cv', 'array'), 'bytes', 'base64', and 'PIL'.
    
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
             shape: list[int, int]|tuple[int, int]|None=None,
             scale: float|None=None,
             shortest: int|None=None,
             longest: int|None=None,
             resample: object=PIL.Image.BILINEAR) -> PIL.Image.Image:
    """Resize the image

    Resize the image to the given shape, scale, shortest, or longest side.

    Args:
        im: ImageSourceTypes: Image source, can be a file path, url, bytes, base64, PIL image, or numpy array.
        shape: tuple: The shape of the resized image (height, width).
        scale: float: The scale factor to resize the image.
        shortest: int: The shortest side of the image.
        longest: int: The longest side of the image.
        resample: int: The resampling filter. Default is PIL.Image.BILINEAR.

    Returns:
        PIL.Image.Image: The resized image.

    Examples:
        >>> imresize('image.jpg', shape=(256, 256))
        >>> imresize('image.jpg', scale=0.5)
        >>> imresize('image.jpg', shortest=256)
        >>> imresize('image.jpg', longest=256)    
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
    


def imwrite(im: ImageSourceTypes, filename: str, quality: int=95) -> None:
    """Save image to file

    Args:
        im: ImageSourceTypes: Image source, can be a file path, url, bytes, base64, PIL image, or numpy array.

    Examples:
        >>> imsave(imread('image.jpg'), 'image.jpg')
        >>> imsave(imread('image.jpg'), 'image.jpg', 100)
    """
    im = imread(im)
    im.save(filename, quality=quality)



def imshow(im: ImageSourceTypes|list[ImageSourceTypes], ncol: int|None=None, nrow: int|None=None) -> object:
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
        ext: list[str]: The list of file extensions to search for. Default is ['.jpg', 'jpeg', '.png', '.tiff'].
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
            raise ValueError(f'The input "{source}" is not found or is neither a file nor a directory.')
    
    return im_list








