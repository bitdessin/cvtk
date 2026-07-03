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
    """Load an image from various sources.

    Loads an image from multiple source types (file path, URL, bytes, base64, PIL image, or numpy array)
    and converts it to PIL.Image.Image. Automatically detects source type and handles EXIF orientation.
    For numpy arrays in BGR format (OpenCV style), converts to RGB for PIL.
    
    Args:
        source (str|pathlib.Path|bytes|PIL.Image.Image|np.ndarray): Image source:
            - str: File path, HTTP/HTTPS URL, or base64 data URI (data:image/...)
            - pathlib.Path: File path object
            - bytes|bytearray: Image binary data
            - PIL.Image.Image: Already loaded PIL image (returned as-is)
            - np.ndarray: Numpy array in BGR format (OpenCV convention)
        exif_transpose (bool): If True, correct image orientation using EXIF metadata. Default is True.
        req_timeout (int): Timeout in seconds for HTTP requests. Default is 60.
    
    Returns:
        PIL.Image.Image: Image in RGB format.
    
    Raises:
        ValueError: If source type is unknown or image cannot be loaded.
    
    Examples:
        >>> im = imread('image.jpg')
        >>> im = imread('https://example.com/image.png')
        >>> im = imread(image_bytes)
        >>> im = imread('data:image/jpeg;base64,...')
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
    """Convert image to different format.

    Loads image from any source and converts to requested output format.
    Supports conversion between PIL, OpenCV (BGR), bytes, base64, and grayscale formats.

    Args:
        source (ImageSourceTypes): Image source (file path, URL, bytes, PIL image, numpy array).
        format (str): Target format. Default is 'PIL'. Options:
            - 'PIL': PIL.Image.Image in RGB
            - 'cv2'/'cv'/'array': numpy array in BGR format (OpenCV convention)
            - 'bytes': JPEG binary data
            - 'base64': Base64-encoded JPEG data URI
            - 'gray'/'grey': PIL.Image.Image in grayscale
    
    Returns:
        PIL.Image.Image|np.ndarray|bytes|str: Image in requested format.
    
    Raises:
        ValueError: If format is unsupported.
    
    Examples:
        >>> im = imread('image.jpg')
        >>> cv_array = imconvert(im, 'cv2')  # BGR numpy array
        >>> base64_uri = imconvert(im, 'base64')
        >>> gray = imconvert(im, 'gray')
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
    """Resize an image to specified dimensions or scale.

    Loads image from any source and resizes using one of multiple methods:
    fixed shape, scale factor, or based on shortest/longest side (useful for aspect ratio preservation).

    Args:
        source (ImageSourceTypes): Image source (file path, URL, bytes, PIL image, numpy array).
        shape (tuple[int,int]|list[int,int]|None): Target size as (width, height). Default is None.
        scale (float|None): Scale factor (0.5 = half size, 2.0 = double). Default is None.
        shortest (int|None): Resize so shortest side equals this. Preserves aspect ratio. Default is None.
        longest (int|None): Resize so longest side equals this. Preserves aspect ratio. Default is None.
        resample: Resampling filter. Default is PIL.Image.BILINEAR (high quality, slower).

    Returns:
        PIL.Image.Image: Resized image in RGB.

    Raises:
        ValueError: If none of shape, scale, shortest, or longest is specified.

    Examples:
        >>> imresize('image.jpg', shape=(256, 256))  # Fixed size
        >>> imresize('image.jpg', scale=0.5)  # Half size
        >>> imresize('image.jpg', shortest=256)  # Shortest side = 256, aspect ratio preserved
        >>> imresize('image.jpg', longest=512)  # Longest side = 512, aspect ratio preserved
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
    """Save image to file.

    Loads image from any source and saves it to disk in JPEG format.

    Args:
        source (ImageSourceTypes): Image source (file path, URL, bytes, PIL image, numpy array).
        filename (str): Output file path. Directory created if needed.
        quality (int): JPEG quality (0-100). Higher = better quality, larger file. Default is 95.

    Returns:
        None. Image saved to disk.

    Examples:
        >>> imwrite('https://example.com/photo.png', 'downloaded.jpg')
        >>> imwrite(imread('input.png'), 'output.jpg', quality=80)
    """
    im = imread(source)
    im.save(filename, quality=quality)



def imshow(
    source: ImageSourceTypes|list[ImageSourceTypes],
    ncol: int|None=None,
    nrow: int|None=None
) -> plt.Figure:
    """Display one or more images in subplots.

    Shows image(s) using matplotlib in an interactive window. Automatically arranges
    multiple images in a grid layout. For file sources, displays filename as subplot title.

    Args:
        source (ImageSourceTypes|list[ImageSourceTypes]): Single image or list of images.
            Each can be file path, URL, bytes, PIL image, or numpy array.
        ncol (int|None): Number of columns in grid. If None, auto-calculated for near-square layout. Default is None.
        nrow (int|None): Number of rows in grid. If None, auto-calculated. Default is None.

    Returns:
        matplotlib.pyplot.Figure: The figure object for further customization.

    Examples:
        >>> imshow('image.jpg')
        >>> imshow(['image1.jpg', 'image2.jpg', 'image3.jpg'], ncol=2)
        >>> imshow([imread('a.png'), imread('b.png'), imread('c.png')])
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
    """Find all image files matching extensions in given paths.

    Recursively searches directories for image files with specified extensions.
    Can process single paths or lists of mixed file and directory paths.

    Args:
        source (str|list[str]): File path(s) or directory path(s) to search.
            - str: Single file or directory
            - list[str]: Multiple files/directories
        ext (str|list[str]): File extensions to match (case-insensitive). Default is ['.jpg', '.jpeg', '.png', '.tiff'].
            Can be single string '.jpg' or list ['jpg', '.png'].

    Returns:
        list[str]: List of matching image file paths (absolute paths, sorted).

    Raises:
        ValueError: If source path is not a file or directory.

    Examples:
        >>> imlist('image_dir')
        >>> imlist(['dir1', 'dir2', 'single_image.jpg'])
        >>> imlist('photos', ext=['.png', '.jpg'])
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

