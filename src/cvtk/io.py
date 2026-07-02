import os
import re
import pathlib
import glob
import io
import typing
import base64
import math
import numpy as np
import matplotlib.pyplot as plt
import requests
import PIL
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


ImageSourceTypes = typing.Union[str, pathlib.Path, bytes, PIL.Image.Image, np.ndarray]


def imread(
    source,
    exif_transpose: bool=True,
    req_timeout: int=60
) -> PIL.Image.Image:
    """Open image

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
    

def imconvert(
    source: ImageSourceTypes,
    format: str='PIL'
) -> ImageSourceTypes:
    """Convert image format

    Convert image format from any format to the specific format.

    Args:
        source (ImageSourceTypes): Image source, can be a file path, url, bytes, base64, PIL image, or numpy array.
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

    im = imread(source)

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


def imresize(
    source: ImageSourceTypes,
    shape: list[int, int]|tuple[int, int]|None=None,
    scale: float|None=None,
    shortest: int|None=None,
    longest: int|None=None,
    resample: object=PIL.Image.BILINEAR
) -> PIL.Image.Image:
    """Resize the image

    Resize the image to the given shape, scale, shortest, or longest side.

    Args:
        source: ImageSourceTypes: Image source, can be a file path, url, bytes, base64, PIL image, or numpy array.
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
    im = imread(source)
    
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
    

def imwrite(
    source: ImageSourceTypes,
    filename: str,
    quality: int=95
) -> None:
    """Save image to file

    Args:
        source: ImageSourceTypes: Image source, can be a file path, url, bytes, base64, PIL image, or numpy array.

    Examples:
        >>> imwrite(imread('image.jpg'), 'image.jpg')
        >>> imwrite(imread('image.jpg'), 'image.jpg', 100)
    """
    im = imread(source)
    im.save(filename, quality=quality)



def imshow(
    source: ImageSourceTypes|list[ImageSourceTypes],
    ncol: int|None=None,
    nrow: int|None=None
) -> plt.Figure:
    """Display image using matplotlib.pyplot

    Args:
        source: ImageSourceTypes: Image or list of images to display.
        ncol: int: Number of columns to display the images. Default is None (automatically set).
        nrow: int: Number of rows to display the images. Default is None (automatically set).
    """
    if not isinstance(source, (list, tuple)):
        source = [source]

    # set subplot panels
    if ncol is None and nrow is None:
        ncol = nrow = 1
        if len(source) > 1:
            ncol = math.ceil(math.sqrt(len(source)))
            nrow = math.ceil(len(source) / ncol)
    elif ncol is None:
        ncol = math.ceil(len(source) / nrow)
    elif nrow is None:
        nrow = math.ceil(len(source) / ncol)
    
    plt.figure()

    for i_, im_ in enumerate(source):
        plt.subplot(nrow, ncol, i_ + 1)
        plt.imshow(imread(im_))
        if isinstance(im_, str):
            plt.title(os.path.basename(im_))

    plt.tight_layout()
    plt.show()
    
    return plt.gcf()


def imlist(
    source: str|list[str],
    ext: str|list[str]=['.jpg', '.jpeg', '.png', '.tiff'],
) -> list[str]:
    """List all image files from the given sources

    The function recevies image sources as a file path, directory path, or a list of file and directory paths.
    If the source is a directory, the function will recursively search for image files with the given extensions.

    Args:
        source: str | list[str]: The directory path.
        ext: list[str]: The list of file extensions to search for. Default is ['.jpg', 'jpeg', '.png', '.tiff'].

    Returns:
        list: List of image files in the directory.
    """
    im_list = []
    if isinstance(source, str):
        sources = [source]
    else:
        sources = source
    
    ext = [e.lower() for e in ext]

    for source in sources:
        if os.path.isdir(source):
            for f in sorted(glob.glob(os.path.join(source, '**', '*'), recursive=True)):
                f_ext = os.path.splitext(f.lower())[1]
                if f_ext in ext:
                    im_list.append(f)
        elif os.path.isfile(source):
            f_ext = os.path.splitext(source.lower())[1]
            if f_ext in ext:
                im_list.append(source)
        else:
            raise ValueError(f'The input "{source}" is not found or is neither a file nor a directory.')
    
    return im_list

