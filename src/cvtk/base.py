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


def imread(source: ImageSourceTypes, format: str='PIL', exif_transpose=True, req_timeout=60) -> ImageSourceTypes:
    """Open and convert image format

    This function supports opening image from file, url, bytes, base64, PIL image, or numpy array.
    Image will be transposed based on the EXIF orientation tag if `exif_transpose` is set to True.
    By default, the image is returned as PIL.Image.Image object
    and can be converted to other formats by setting `format` argument.
    Note that, if 'cv2' format is selected, the image will be in BGR format, compatible with OpenCV.
    
    Args:
        source: str | pathlib.Path | bytes | PIL.Image.Image | np.ndarray: Image source,
            can be a file path, url, bytes, base64, PIL image, or numpy array.
        format: str: The format of the returned image. Default is 'PIL'.
            Options are 'cv2', 'bytes', 'base64', 'PIL'.
        exif_transpose: bool: Whether to transpose the image based on the EXIF orientation tag.
        req_timeout: int: The timeout for the request to get image from url. Default is 60 seconds.
    
    Returns:
        ImageSourceTypes: Image data in the specified format.
        
    Examples:
        >>> imread('image.jpg')
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
        im[..., :3] = im[..., 2::-1]
        im = source
    
    else:
        raise ValueError(f'Unable open image file due to unknown type of "{source}".')
    
    if im is None:
        raise ValueError(f'Unable open image file f{source}. Check if the file exists or the url is correct.')

    return __imconvert(im, format)
    



def imconvert(im: ImageSourceTypes, format: str='PIL') -> ImageSourceTypes:
    """Convert image format

    Convert image format from any format to the specific format.

    Args:
        im: ImageSourceTypes: Image data in numpy array format.
        format: str: The format of the returned image. Default is 'PIL'.
            Options are 'cv2', 'bytes', 'base64', 'PIL'.
    
    Returns:
        ImageSourceTypes: Image data in the specified format.
        
    Examples:
        >>> im = imread('image.jpg')
        >>> imconvert(im, 'cv2')
    """
    return imread(im, format)



def __imconvert(im, format):
    if format.lower() in ['array', 'cv2', 'cv']:
        return __pil2cv(im)
    elif format.lower() == 'pil':
        return im
    elif format.lower() == 'bytes':
        return __pil2bytes(im)
    elif format.lower() == 'base64':
        return base64.b64encode(__pil2bytes(im)).decode('utf-8')
    elif format.lower() in ['gray', 'grey']:
        return __pil2gray(im)
    else:
        raise ValueError(f'Unsupported image format "{format}".')




def __pil2cv(im: PIL.Image.Image) -> np.ndarray:
    return np.array(im)[..., 2::-1]




def __pil2bytes(im: PIL.Image.Image, format='jpg') -> bytes:
    im_buff = io.BytesIO()
    im.save(im_buff, format=format)
    return im_buff.getvalue()
    


def __pil2gray(im: PIL.Image.Image) -> PIL.Image.Image:
    return im.convert('L')



def imresize(im: ImageSourceTypes, shape=None, scale=None, shortest=None, longest=None, resample=PIL.Image.BILINEAR) -> PIL.Image.Image:
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
    im = imread(im, format='PIL')
    
    if shape is not None:
        return im.resize(shape, resample=resample)
    elif scale is not None:
        return im.resize((int(im.width * scale), int(im.height * scale)), resample=resample)
    elif shortest is not None:
        ratio = shortest / min(im.size)
        return im.resize((int(im.width * ratio), int(im.height * ratio)), resample=resample)
    elif longest is not None:
        ratio = longest / max(im.size)
        return im.resize((int(im.width * ratio), int(im.height * ratio)), resample=resample)
    else:
        raise ValueError('Specify the shape, scale, shortest, or longest side to resize the image.')
    


def imwrite(im: ImageSourceTypes, output: str, quality: int=95):
    """Save image to file

    Args:
        im: ImageSourceTypes: Image data in numpy array format.

    Examples:
        >>> imsave(imread('image.jpg'), 'image.jpg')
        >>> imsave(imread('image.jpg'), 'image.jpg', 100)
    """
    im = imread(im, format='PIL')
    im.save(output, quality=quality)



def imshow(im: ImageSourceTypes | list[ImageSourceTypes], ncol: int|None=None, nrow: int|None=None):
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
                          'Install matplotlib package to enable this feature.') from e

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
    
    for i_, im_ in enumerate(im):
        plt.subplot(nrow, ncol, i_ + 1)
        plt.imshow(imread(im_, format='PIL'))
        if isinstance(im_, str):
            plt.title(os.path.basename(im_))

    plt.show()




def imlist(source: str | list[str], ext: list[str]=['.jpg', '.jpeg', '.png', '.tiff']) -> list[str]:
    """List all image files from the given sources

    The function recevies image sources as a file path, directory path, or a list of file and directory paths.
    If the source is a directory, the function will recursively search for image files with the given extensions.

    Args:
        source: str | list[str]: The directory path.
        ext: list[str]: The list of file extensions to search for. Default is ['.jpg', '.png', '.tiff'].

    Returns:
        list: List of image files in the directory.
    """
    im_list = []
    sources = [source] if isinstance(source, str) else source

    for source in sources:
        if isinstance(source, list):
            im_list.append(imlist(source, ext))
        else:
            if os.path.isdir(source):
                for f in glob.glob(os.path.join(source, '**', '*'), recursive=True):
                    if os.path.splitext(f)[1].lower() in ext:
                        im_list.append(f)
            elif os.path.isfile(source):
                im_list.append(source)
            else:
                raise ValueError(f'The given "{source}" is neither a file nor a directory.')
                
    return im_list




class __JsonEncoder(json.JSONEncoder):
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
            return super(__JsonEncoder, self).default(obj)