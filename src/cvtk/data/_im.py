from __future__ import annotations
from dataclasses import dataclass, field

import os
import pathlib
import random
import typing
import copy
import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image
import PIL.ImageFile
import PIL.ImageDraw
import PIL.ImageFont
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

import skimage
import skimage.measure

import pycocotools
import pycocotools.mask

import cvtk


_MaskArray = npt.NDArray[np.bool_]
_RLE = dict[str, typing.Any]
_Polygons = list[list[float]]
_ImgSize = tuple[int, int]


def _clip_coords(
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    imsize: _ImgSize
) -> tuple[float, float, float, float]:
    x_min = max(0.0, min(x_min, imsize[0]))
    y_min = max(0.0, min(y_min, imsize[1]))
    x_max = max(0.0, min(x_max, imsize[0]))
    y_max = max(0.0, min(y_max, imsize[1]))
    return y_min, x_min, y_max, x_max


@dataclass(frozen=True)
class Bbox:
    """Bounding box representation with multiple formats and conversions.
    
    Supports multiple coordinate formats (pixel and normalized) and provides
    conversions between them. All pixel coordinates are clipped to image boundaries.
    
    Args:
        y_min (float | None): Minimum y coordinate (top). Default None.
        x_min (float | None): Minimum x coordinate (left). Default None.
        y_max (float | None): Maximum y coordinate (bottom). Default None.
        x_max (float | None): Maximum x coordinate (right). Default None.
        imsize (tuple[int, int] | None): Image size as (width, height). Default None.
    """
    y_min: float | None = field(default=None, repr=False)
    x_min: float | None = field(default=None, repr=False)
    y_max: float | None = field(default=None, repr=False)
    x_max: float | None = field(default=None, repr=False)
    imsize: _ImgSize | None = field(default=None, repr=False)    
    
    
    @classmethod
    def from_xywh(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        imsize: _ImgSize,
    ) -> Bbox:
        """Create bbox from top-left corner (x, y) and width, height.

        Args:
            x (float): Top-left x coordinate (pixel units).
            y (float): Top-left y coordinate (pixel units).
            w (float): Bounding box width (pixel units).
            h (float): Bounding box height (pixel units).
            imsize (tuple[int, int]): Image size as (width, height). Required for clipping.

        Returns:
            Bbox: Bounding box object with coordinates clipped to image boundaries.

        Examples:
            >>> bbox = Bbox.from_xywh(10, 20, 100, 50, imsize=(640, 480))
            >>> bbox.to_xyxy()
            (10, 20, 110, 70)
        """
        y_min, x_min, y_max, x_max = _clip_coords(y, x, y + h, x + w, imsize)
        return cls(y_min, x_min, y_max, x_max, imsize)


    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        imsize: _ImgSize,
    ) -> Bbox:
        """Create bbox from top-left and bottom-right corners.

        Args:
            x1 (float): Top-left x coordinate (pixel units).
            y1 (float): Top-left y coordinate (pixel units).
            x2 (float): Bottom-right x coordinate (pixel units).
            y2 (float): Bottom-right y coordinate (pixel units).
            imsize (tuple[int, int]): Image size as (width, height). Required for clipping.

        Returns:
            Bbox: Bounding box object with coordinates clipped to image boundaries.

        Examples:
            >>> bbox = Bbox.from_xyxy(10, 20, 110, 70, imsize=(640, 480))
            >>> bbox.width
            100.0
        """
        y_min, x_min, y_max, x_max = _clip_coords(y1, x1, y2, x2, imsize)
        return cls(y_min, x_min, y_max, x_max, imsize)


    @classmethod
    def from_cxcywh(
        cls,
        cx: float,
        cy: float,
        w: float,
        h: float,
        imsize: _ImgSize,
    ) -> Bbox:
        """Create bbox from center coordinates and width, height.

        Args:
            cx (float): Center x coordinate (pixel units).
            cy (float): Center y coordinate (pixel units).
            w (float): Bounding box width (pixel units).
            h (float): Bounding box height (pixel units).
            imsize (tuple[int, int]): Image size as (width, height). Required for clipping.

        Returns:
            Bbox: Bounding box object with coordinates clipped to image boundaries.

        Examples:
            >>> bbox = Bbox.from_cxcywh(60, 45, 100, 50, imsize=(640, 480))
            >>> bbox.to_cxcywh()
            (60.0, 45.0, 100.0, 50.0)
        """
        half_w = w / 2.0
        half_h = h / 2.0
        y_min, x_min, y_max, x_max = _clip_coords(cy - half_h, cx - half_w, cy + half_h, cx + half_w, imsize)
        return cls(y_min, x_min, y_max, x_max, imsize)

    
    @classmethod
    def from_xyxyr(
        cls,
        x1r: float,
        y1r: float,
        x2r: float,
        y2r: float,
        imsize: _ImgSize,
    ) -> Bbox:
        """Create bbox from normalized top-left and bottom-right corners in range [0, 1].

        Args:
            x1r (float): Normalized top-left x coordinate [0, 1].
            y1r (float): Normalized top-left y coordinate [0, 1].
            x2r (float): Normalized bottom-right x coordinate [0, 1].
            y2r (float): Normalized bottom-right y coordinate [0, 1].
            imsize (tuple[int, int]): Image size as (width, height). Required for denormalization and clipping.

        Returns:
            Bbox: Bounding box object with pixel coordinates.

        Examples:
            >>> bbox = Bbox.from_xyxyr(0.0, 0.0, 0.5, 0.5, imsize=(640, 480))
            >>> bbox.to_xyxy()
            (0.0, 0.0, 320.0, 240.0)
        """
        x1 = x1r * imsize[0]
        y1 = y1r * imsize[1]
        x2 = x2r * imsize[0]
        y2 = y2r * imsize[1]
        y_min, x_min, y_max, x_max = _clip_coords(y1, x1, y2, x2, imsize)
        return cls(y_min, x_min, y_max, x_max, imsize)

    
    @classmethod
    def from_xywhr(
        cls,
        xr: float,
        yr: float,
        wr: float,
        hr: float,
        imsize: _ImgSize,
    ) -> Bbox:
        """Create bbox from normalized top-left corner and dimensions in range [0, 1].

        Args:
            xr (float): Normalized top-left x coordinate [0, 1].
            yr (float): Normalized top-left y coordinate [0, 1].
            wr (float): Normalized width [0, 1].
            hr (float): Normalized height [0, 1].
            imsize (tuple[int, int]): Image size as (width, height). Required for denormalization and clipping.

        Returns:
            Bbox: Bounding box object with pixel coordinates.

        Examples:
            >>> bbox = Bbox.from_xywhr(0.0, 0.0, 0.5, 0.5, imsize=(640, 480))
            >>> bbox.width
            320.0
        """
        x = xr * imsize[0]
        y = yr * imsize[1]
        w = wr * imsize[0]
        h = hr * imsize[1]
        y_min, x_min, y_max, x_max = _clip_coords(y, x, y + h, x + w, imsize)
        return cls(y_min, x_min, y_max, x_max, imsize)

    
    @classmethod
    def from_cxcywhr(
        cls,
        cxr: float,
        cyr: float,
        wr: float,
        hr: float,
        imsize: _ImgSize,
    ) -> Bbox:
        """Create bbox from normalized center and dimensions in range [0, 1].

        Args:
            cxr (float): Normalized center x coordinate [0, 1].
            cyr (float): Normalized center y coordinate [0, 1].
            wr (float): Normalized width [0, 1].
            hr (float): Normalized height [0, 1].
            imsize (tuple[int, int]): Image size as (width, height). Required for denormalization and clipping.

        Returns:
            Bbox: Bounding box object with pixel coordinates.

        Examples:
            >>> bbox = Bbox.from_cxcywhr(0.25, 0.25, 0.5, 0.5, imsize=(640, 480))
            >>> bbox.to_cxcywh()
            (80.0, 60.0, 320.0, 240.0)
        """
        cx = cxr * imsize[0]
        cy = cyr * imsize[1]
        w = wr * imsize[0]
        h = hr * imsize[1]
        half_w = w / 2.0
        half_h = h / 2.0
        y_min, x_min, y_max, x_max = _clip_coords(cy - half_h, cx - half_w, cy + half_h, cx + half_w, imsize)
        return cls(y_min, x_min, y_max, x_max, imsize)


    @property
    def width(self) -> float:
        """Width of the bounding box in pixels.

        Returns:
            float: Width calculated as x_max - x_min.
        """
        
        return self.x_max - self.x_min


    @property
    def height(self) -> float:
        """Height of the bounding box in pixels.

        Returns:
            float: Height calculated as y_max - y_min.
        """
        return self.y_max - self.y_min


    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels.

        Returns:
            float: Area calculated as width × height.
        """
        return float(self.width * self.height)


    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format.

        Returns:
            tuple[float, float, float, float]: Top-left (x1, y1) and bottom-right (x2, y2) coordinates.
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)


    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, w, h) format.

        Returns:
            tuple[float, float, float, float]: Top-left (x, y) and dimensions (width, height).
        """
        return (self.x_min, self.y_min, self.width, self.height)
    
    
    def to_cxcywh(self) -> tuple[float, float, float, float]:
        """Convert to (cx, cy, w, h) format.

        Returns:
            tuple[float, float, float, float]: Center (cx, cy) and dimensions (width, height).
        """
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        return (cx, cy, self.width, self.height)

    
    def to_xyxyr(self) -> tuple[float, float, float, float]:
        """Convert to normalized (x1, y1, x2, y2) format in range [0, 1].

        Returns:
            tuple[float, float, float, float]: Normalized coordinates [0, 1].
        """
        x1r = self.x_min / self.imsize[0]
        y1r = self.y_min / self.imsize[1]
        x2r = self.x_max / self.imsize[0]
        y2r = self.y_max / self.imsize[1]
        return (x1r, y1r, x2r, y2r)

    
    def to_xywhr(self) -> tuple[float, float, float, float]:
        """Convert to normalized (x, y, w, h) format in range [0, 1].

        Returns:
            tuple[float, float, float, float]: Normalized coordinates and dimensions [0, 1].
        """
        xr = self.x_min / self.imsize[0]
        yr = self.y_min / self.imsize[1]
        wr = self.width / self.imsize[0]
        hr = self.height / self.imsize[1]
        return (xr, yr, wr, hr)
    
    
    def to_cxcywhr(self) -> tuple[float, float, float, float]:
        """Convert to normalized (cx, cy, w, h) format in range [0, 1].

        Returns:
            tuple[float, float, float, float]: Normalized center and dimensions [0, 1].
        """
        cxr = (self.x_min + self.x_max) / 2.0 / self.imsize[0]
        cyr = (self.y_min + self.y_max) / 2.0 / self.imsize[1]
        wr = self.width / self.imsize[0]
        hr = self.height / self.imsize[1]
        return (cxr, cyr, wr, hr)
    
    
    @staticmethod
    def xyxy2xywh(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x1, y1, x2, y2) to (x, y, w, h).

        Args:
            xyxy (tuple[float, float, float, float]): Corner coordinates (x1, y1, x2, y2).

        Returns:
            tuple[float, float, float, float]: Top-left and dimensions (x, y, w, h).
        """
        x1, y1, x2, y2 = xyxy
        return (x1, y1, x2 - x1, y2 - y1)
    
    
    @staticmethod
    def xywh2xyxy(xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x, y, w, h) to (x1, y1, x2, y2).

        Args:
            xywh (tuple[float, float, float, float]): Top-left and dimensions (x, y, w, h).

        Returns:
            tuple[float, float, float, float]: Corner coordinates (x1, y1, x2, y2).
        """
        x, y, w, h = xywh
        return (x, y, x + w, y + h)

    
    @staticmethod
    def xywh2cxcywh(xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x, y, w, h) to (cx, cy, w, h).

        Args:
            xywh (tuple[float, float, float, float]): Top-left and dimensions (x, y, w, h).

        Returns:
            tuple[float, float, float, float]: Center and dimensions (cx, cy, w, h).
        """
        x, y, w, h = xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        return (cx, cy, w, h)

    
    @staticmethod
    def cxcywh2xywh(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (cx, cy, w, h) to (x, y, w, h).

        Args:
            cxcywh (tuple[float, float, float, float]): Center and dimensions (cx, cy, w, h).

        Returns:
            tuple[float, float, float, float]: Top-left and dimensions (x, y, w, h).
        """
        cx, cy, w, h = cxcywh
        x = cx - w / 2.0
        y = cy - h / 2.0
        return (x, y, w, h)

    
    @staticmethod
    def xyxy2cxcywh(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x1, y1, x2, y2) to (cx, cy, w, h).

        Args:
            xyxy (tuple[float, float, float, float]): Corner coordinates (x1, y1, x2, y2).

        Returns:
            tuple[float, float, float, float]: Center and dimensions (cx, cy, w, h).
        """
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)

    
    @staticmethod
    def cxcywh2xyxy(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2).

        Args:
            cxcywh (tuple[float, float, float, float]): Center and dimensions (cx, cy, w, h).

        Returns:
            tuple[float, float, float, float]: Corner coordinates (x1, y1, x2, y2).
        """
        cx, cy, w, h = cxcywh
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        return (x1, y1, x2, y2)
    
    
    @staticmethod
    def xyxyr2xywhr(xyxyr: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert normalized (x1, y1, x2, y2) to normalized (x, y, w, h).

        Args:
            xyxyr (tuple[float, float, float, float]): Normalized corner coordinates [0, 1].

        Returns:
            tuple[float, float, float, float]: Normalized top-left and dimensions [0, 1].
        """
        x1r, y1r, x2r, y2r = xyxyr
        return (x1r, y1r, x2r - x1r, y2r - y1r)
    
    
    @staticmethod
    def xywhr2xyxyr(xywhr: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert normalized (x, y, w, h) to normalized (x1, y1, x2, y2).

        Args:
            xywhr (tuple[float, float, float, float]): Normalized top-left and dimensions [0, 1].

        Returns:
            tuple[float, float, float, float]: Normalized corner coordinates [0, 1].
        """
        xr, yr, wr, hr = xywhr
        return (xr, yr, xr + wr, yr + hr)


@dataclass
class Segm:
    """Segmentation representation supporting mask, RLE, and polygon formats.
    
    Flexible representation that can internally store segmentation in any format
    (binary mask, RLE, or polygons) and convert on-demand. Only one format is required
    at construction; others are generated lazily.
    
    Args:
        _mask (np.ndarray | None): Binary mask array of shape (height, width). Default None.
        _rle (dict | None): COCO RLE format with 'counts' and 'size' keys. Default None.
        _polygons (list[list[float]] | None): List of polygon coordinates. Default None.
        _imsize (tuple[int, int] | None): Image size as (width, height). Default None.
    
    Raises:
        ValueError: If no segmentation format provided or mask shape doesn't match imsize.
    """
    _mask: _MaskArray | None = field(default=None, repr=False)
    _rle: _RLE | None = field(default=None, repr=False)
    _polygons: _Polygons | None = field(default=None, repr=False)
    _imsize: _ImgSize | None = field(default=None, repr=False)


    def __post_init__(self) -> None:
        n_sources = sum(
            x is not None
            for x in (self._mask, self._rle, self._polygons)
        )
        if n_sources == 0:
            raise ValueError("mask, rle, or polygons must be provided.")

        if self._mask is not None:
            self._mask = np.asarray(self._mask, dtype=bool)
            if self._mask.shape != (self._imsize[1], self._imsize[0]):
                raise ValueError(f"mask shape must be {(self._imsize[1], self._imsize[0])}, but got {self._mask.shape}.")

        if self._rle is not None:
            _rel = dict(self._rle)
            if self._imsize is not None:
                _rel["size"] = [self._imsize[1], self._imsize[0]]
            self._rle = _rel
        
        if self._polygons is not None:
            self._polygons = [list(p) for p in self._polygons]


    @classmethod
    def from_mask(
        cls,
        mask: np.typing.ArrayLike
    ) -> Segm:
        """Create Segm from a binary mask array.
        
        Args:
            mask (np.typing.ArrayLike): Binary mask array of shape (height, width).
            
        Returns:
            Segm: Segmentation object initialized with mask.
            
        Examples:
            >>> mask = np.random.rand(100, 100) > 0.5
            >>> segm = Segm.from_mask(mask)
            >>> segm.area
            5000.0
        """
        mask_array = np.asarray(mask, dtype=bool)
        return cls(_mask=mask_array, _imsize=(mask_array.shape[1], mask_array.shape[0]))


    @classmethod
    def from_rle(
        cls,
        rle: _RLE,
        *,
        imsize: _ImgSize | None = None,
    ) -> Segm:
        """Create Segm from RLE representation.
        
        Args:
            rle (dict): COCO RLE format dictionary with 'counts' and optional 'size' keys.
            imsize (tuple[int, int] | None): Image size (width, height). Required if RLE has no size. Default None.
            
        Returns:
            Segm: Segmentation object initialized with RLE.
            
        Raises:
            ValueError: If neither RLE nor imsize provides image dimensions.
            
        Examples:
            >>> rle = {'size': [100, 100], 'counts': b'...'}
            >>> segm = Segm.from_rle(rle)
        """
        
        rel_size = rle.get("size")
        if rel_size is not None:
            # RLE size is [height, width], but imsize should be (width, height)
            imsize = imsize or (rel_size[1], rel_size[0])

        if imsize is None:
            raise ValueError("imsize is required when rle has no size.")

        # Normalize RLE dict: convert bytes in 'counts' to string for JSON serialization
        normalized_rle = dict(rle)
        if isinstance(normalized_rle.get("counts"), bytes):
            normalized_rle["counts"] = normalized_rle["counts"].decode("ascii")
        
        return cls(_rle=normalized_rle, _imsize=imsize)


    @classmethod
    def from_polygons(
        cls,
        polygons: _Polygons,
        *,
        imsize: _ImgSize,
    ) -> Segm:
        """Create Segm from polygon representation.
        
        Args:
            polygons (list[list[float]]): List of polygons, each a flat list of (x, y) coordinates.
            imsize (tuple[int, int]): Image size (width, height). Required.
            
        Returns:
            Segm: Segmentation object initialized with polygons.
            
        Raises:
            ValueError: If imsize is not provided.
            
        Examples:
            >>> polygons = [[10, 10, 100, 10, 100, 100, 10, 100]]
            >>> segm = Segm.from_polygons(polygons, imsize=(256, 256))
        """
        return cls(_polygons=[list(poly) for poly in polygons], _imsize=imsize)


    @property
    def size(self) -> _ImgSize:
        """Image size (width, height) associated with the segmentation.
        
        Returns:
            tuple[int, int]: Image size as (width, height).
        """
        return self._imsize
    
    
    @property
    def area(self) -> float:
        """Area of the segmentation in pixels.
        
        Returns:
            float: Number of pixels where mask is True.
        """
        return float(np.sum(self.to_mask()))


    def to_mask(
        self,
    ) -> _MaskArray:
        """Convert to binary mask representation.
        
        Returns:
            np.ndarray: Binary mask of shape (height, width) with dtype bool.
            
        Examples:
            >>> segm = Segm.from_mask(mask)
            >>> converted_mask = segm.to_mask()
        """
        if self._mask is None:
            if self._rle is not None:
                self._mask = self._rle_to_mask(self._rle)
            elif self._polygons is not None:
                self._mask = self._polygons_to_mask(self._polygons)
            else:
                raise RuntimeError("Segm has no valid representation.")
        return copy.deepcopy(self._mask)


    def to_rle(
        self,
        *,
        compressed: bool = True
    ) -> _RLE:
        """Convert to RLE representation.
        
        Args:
            compressed (bool): Use compressed format. Uncompressed not supported. Default True.
            
        Returns:
            dict: COCO RLE format with 'size' and 'counts' keys.
            
        Raises:
            NotImplementedError: If compressed=False.
        """
        if self._rle is None:
            self._rle = self._mask_to_rle(self.to_mask(), compressed=compressed)
        return copy.deepcopy(self._rle)


    def to_polygons(
        self
    ) -> _Polygons:
        """Convert to polygon representation.
        
        Returns:
            list[list[float]]: List of polygons, each a flat list of (x, y) coordinates.
            
        Examples:
            >>> segm = Segm.from_mask(mask)
            >>> polygons = segm.to_polygons()
        """
        if self._polygons is None:
            self._polygons = self._mask_to_polygons(self.to_mask())
        polygons = [poly.copy() for poly in self._polygons]
        return copy.deepcopy(polygons)
    

    def _mask_to_rle(self, mask: _MaskArray, *, compressed: bool = True) -> _RLE:
        rle = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        
        if not compressed:
            raise NotImplementedError("Uncompressed RLE is not supported.")
        
        return {
            "size": [self._imsize[1], self._imsize[0]],  # [height, width]
            "counts": rle["counts"],
        }


    def _rle_to_mask(self, rle: _RLE) -> _MaskArray:
        rle = dict(rle)

        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")

        mask = pycocotools.mask.decode(rle)

        return mask.astype(bool)


    def _polygons_to_mask(self, polygons: _Polygons) -> _MaskArray:
        rles = pycocotools.mask.frPyObjects(polygons, self._imsize[1], self._imsize[0])
        rle = pycocotools.mask.merge(rles)
        mask = pycocotools.mask.decode(rle)
        return mask.astype(bool)


    def _mask_to_polygons(self, mask: _MaskArray) -> _Polygons:
        polygons: list[list[float]] = []

        contours = skimage.measure.find_contours(mask.astype(np.uint8), 0.5)

        for contour in contours:
            if len(contour) < 3:
                continue

            polygon: list[float] = []

            for y, x in contour:
                polygon.extend([float(x), float(y)])

            if len(polygon) >= 6:
                polygons.append(polygon)

        return polygons



    def to_dict(
        self,
        *,
        format: str = "rle"
    ) -> dict[str, typing.Any]:
        """Convert Segm to a dictionary representation in the specified format.
        
        Args:
            format (str): Output format: 'mask', 'rle', 'polygon', or 'polygons'. Default 'rle'.
            
        Returns:
            dict: Dictionary with keys 'width', 'height', 'area', and format-specific key.
            
        Raises:
            ValueError: If format not in ['mask', 'rle', 'polygon', 'polygons'].
            
        Examples:
            >>> segm = Segm.from_mask(mask)
            >>> d = segm.to_dict(format='rle')
            >>> d.keys()
            dict_keys(['width', 'height', 'area', 'rle'])
        """    
        data: dict[str, typing.Any] = {
            "width": self._imsize[0],
            "height": self._imsize[1],
            "area": self.area,
        }
        if format == "mask":
            data["mask"] = self.to_mask().astype(np.uint8).tolist()
        elif format == "rle":
            data["rle"] = self.to_rle()
        elif format in {"polygon", "polygons"}:
            data["polygons"] = self.to_polygons()
        else:
            raise ValueError(f"Unsupported segmentation format: {format}")

        return data


@dataclass
class InstanceAnnotation:
    """Represents an instance annotation with label, bounding box, segmentation, and score.
    
    Combines various annotation types (label, bbox, segmentation) with optional confidence score.
    Provides conversion to dictionary representations with flexible format selection.
    
    Args:
        label (str): Class label name.
        bbox (Bbox | None): Bounding box annotation. Default None.
        segm (Segm | None): Segmentation annotation. Default None.
        score (float | None): Confidence score [0, 1]. Default None.
    """
    label: str
    bbox: Bbox | None = None
    segm: Segm | None = None
    score: float | None = None
 
    @property
    def area(self) -> float | None:
        """Area of the annotation in pixels.
        
        Returns from segmentation if available, otherwise from bounding box.
        Returns None if neither bbox nor segm are present.
        
        Returns:
            float | None: Area in square pixels or None.
        """
        
        if self.segm is not None:
            return self.segm.area
        if self.bbox is not None:
            return self.bbox.area
        return None
    

    def to_dict(
        self,
        *,
        bbox_format: str = "xyxy",
        segm_format: str = "rle"
    ) -> dict[str, typing.Any]:
        """Convert InstanceAnnotation to a dictionary representation with specified formats.
        
        Args:
            bbox_format (str): Bounding box format: 'xyxy', 'xywh', or 'cxcywh'. Default 'xyxy'.
            segm_format (str): Segmentation format: 'rle', 'polygon'/'polygons', or 'mask'. Default 'rle'.
            
        Returns:
            dict: Dictionary with keys 'label', 'bbox', 'segm', 'score', 'area'.
            
        Raises:
            ValueError: If bbox_format or segm_format not supported.
            
        Examples:
            >>> ann = InstanceAnnotation(label='cat', bbox=bbox, score=0.95)
            >>> d = ann.to_dict()
            >>> d['label']
            'cat'
        """
        if bbox_format == "xyxy":
            bbox = self.bbox.to_xyxy() if self.bbox is not None else None
        elif bbox_format == "xywh":
            bbox = self.bbox.to_xywh() if self.bbox is not None else None
        elif bbox_format == "cxcywh":
            bbox = self.bbox.to_cxcywh() if self.bbox is not None else None
        else:
            raise ValueError(f"Unsupported bbox format: {bbox_format}")
        
        if segm_format == 'rle':
            segm = self.segm.to_rle() if self.segm is not None else None
        elif segm_format in {'polygon', 'polygons'}:
            segm = self.segm.to_polygons() if self.segm is not None else None
        elif segm_format == 'mask':
            segm = self.segm.to_mask() if self.segm is not None else None
        else:
            raise ValueError(f"Unsupported segmentation format: {segm_format}")
        
        return {
            "label": self.label,
            "bbox": bbox,
            "segm": segm,
            "score": self.score,
            "area": self.area,
        }


@dataclass
class ImageRecord:
    """Represents an image record with source path and instance annotations.
    
    Manages a single image with its associated annotations (bounding boxes, segmentations).
    Validates all annotations against image dimensions. Supports drawing annotations and
    serialization to various formats.
    
    Args:
        source (str | os.PathLike): Path to image file.
        annotations (list[InstanceAnnotation]): List of annotations. Default empty list.
        size (tuple[int, int] | None): Image dimensions as (width, height). If provided, skips reading file.
        
    Attributes:
        source (pathlib.Path): Image file path.
        size (tuple[int, int]): Image dimensions as (width, height).
        annotations (list[InstanceAnnotation]): List of annotations.
    """
    source: str | os.PathLike[str]
    annotations: list[InstanceAnnotation] = field(default_factory=list)
    size: tuple[int, int] | None = None


    def __post_init__(self) -> None:
        # If size is provided, skip reading the file
        if self.size is None:
            im = cvtk.io.imread(self.source)
            self.size = im.size

        self.source = pathlib.Path(self.source)
        self.annotations = list(self.annotations)

        self._validate_annotations()

    def add_annotation(self, annotation: InstanceAnnotation) -> None:
        """Add an InstanceAnnotation to the ImageRecord after validation.
        
        Args:
            annotation (InstanceAnnotation): Annotation to add.
            
        Raises:
            TypeError: If annotation is not InstanceAnnotation.
            ValueError: If annotation coordinates are outside image or bbox has zero area.
        """
        if not isinstance(annotation, InstanceAnnotation):
            raise TypeError("annotation must be an InstanceAnnotation.")

        self._validate_annotation(annotation)
        self.annotations.append(annotation)


    def _validate_annotations(self) -> None:
        """Validate all annotations in the ImageRecord."""
        for annotation in self.annotations:
            self._validate_annotation(annotation)


    def _validate_annotation(self, annotation: InstanceAnnotation) -> None:
        if annotation.bbox is not None:
            x1, y1, x2, y2 = annotation.bbox.to_xyxy()
            if x1 < 0 or y1 < 0:
                print(self.source)
                print(self.size)
                print([x1, y1, x2, y2])
                raise ValueError("bbox coordinates must be non-negative.")
            if x2 > self.size[0] or y2 > self.size[1]:
                print(self.source)
                print(self.size)
                print([x1, y1, x2, y2])
                
                raise ValueError("bbox is outside the image.")
            
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must have positive width and height.")

        if annotation.segm is not None:
            if annotation.segm.to_mask().shape != (self.size[1], self.size[0]):
                print(self.source)
                print(self.size)
                print(annotation.segm.to_mask().shape)
                
                raise ValueError(f"mask shape must be {(self.size[1], self.size[0])}, got {annotation.segm.to_mask().shape}.")
    

    def to_dict(self) -> dict[str, typing.Any]:
        """Convert ImageRecord to a dictionary representation.
        
        Returns:
            dict: Dictionary with keys 'source', 'size', 'annotations'.
            
        Examples:
            >>> record = ImageRecord(source='image.jpg')
            >>> d = record.to_dict()
            >>> d.keys()
            dict_keys(['source', 'size', 'annotations'])
        """
        return {
            "source": str(self.source),
            "size": self.size,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }


    def draw(
        self,
        layers: str | list[str] = 'bbox',
        output: str | os.PathLike[str] | None = None,
        cutoff: float = 0.0,
        label: bool = True,
        score: bool = True,
        font: PIL.ImageFont.ImageFont | None = None,
        colors: dict[str, list[int, int, int]] | None = None,
    ) -> PIL.Image.Image:
        """Draw annotations on an image.
        
        Visualizes bounding boxes, segmentation masks, and labels on the image.
        Supports multiple rendering layers that can be combined.
        
        Args:
            layers (str|list[str]): Layers to draw. Valid: 'bbox', 'segm', 'mask', 'overlay'. Default 'bbox'.
            output (str|os.PathLike|None): Save drawn image to this path. Default None (no save).
            cutoff (float): Confidence cutoff. Skip annotations with score < cutoff. Default 0.0.
            label (bool): Draw class label text. Default True.
            score (bool): Draw confidence score with label. Default True.
            font (PIL.ImageFont|None): Font for text. If None, default font used. Default None.
            colors (dict|None): Map label names to RGB tuples (0-255). If None, random colors. Default None.
            
        Returns:
            PIL.Image.Image: Drawn image in RGB mode.
            
        Raises:
            ValueError: If layer names not valid.
            
        Examples:
            >>> record = ImageRecord(source='image.jpg')
            >>> record.add_annotation(ann)
            >>> drawn = record.draw(layers=['bbox'], output='drawn.jpg')
            >>> drawn = record.draw(layers=['mask', 'overlay'], cutoff=0.5)
        """
        def get_color(col_dict: dict[str, tuple], label_name: str | None):
            key = label_name or "___UNDEF___"
            if key not in col_dict:
                col_dict[key] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            return col_dict[key]
        
        
        layers = cvtk.utils.as_list(layers)
        if set(layers) - {"bbox", "segm", "mask", "overlay"}:
            raise ValueError(f"Valid layers are: 'bbox', 'segm', 'mask', and 'overlay', but got: {layers}")
    
        im = cvtk.io.imread(self.source).convert("RGB")
        draw_ctx = PIL.ImageDraw.Draw(im)

        # font
        if font is None:
            font_size = max(10, int(min(im.width, im.height) / 50))
            try:
                font = PIL.ImageFont.load_default(size=font_size)
            except TypeError:
                font = PIL.ImageFont.load_default()
        
        # colors
        colors = dict(colors) if colors is not None else {}
        
        # line width
        outline_width = max(2, int(min(im.width, im.height) / 200))

        # setup colored mask image (black background, colored objects)
        if 'mask' in layers or 'overlay' in layers:
            mask_image = np.zeros((im.height, im.width, 3), dtype=np.uint8)
            for ann in self.annotations:
                if (ann.score is not None) and (ann.score < cutoff):
                    continue
                col = get_color(colors, ann.label)
                if ann.segm is not None:
                    segm_mask = ann.segm.to_mask()
                    mask_image[segm_mask > 0] = col
            if 'mask' in layers:
                im = PIL.Image.fromarray(mask_image.astype(np.uint8))
        else:
            mask_image = None
        
        # draw annotations
        for ann in self.annotations:
            if (ann.score is not None) and (ann.score < cutoff):
                continue
            
            col = get_color(colors, ann.label)
            label_x, label_y = None, None

            # draw segmentation polygons
            if "segm" in layers and ann.segm is not None:
                for polygon in ann.segm.to_polygons():
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    if len(points) >= 2:
                        draw_ctx.line(points, fill=col, width=outline_width)
                    if label_x is None or label_x < min(p[0] for p in points):
                        label_x = min(p[0] for p in points)
                    if label_y is None or label_y < min(p[1] for p in points):
                        label_y = min(p[1] for p in points)                

            # draw bounding box
            if "bbox" in layers and ann.bbox is not None:
                x1, y1, x2, y2 = ann.bbox.to_xyxy()
                draw_ctx.rectangle([(x1, y1), (x2, y2)], outline=col, width=outline_width)
                label_x, label_y = x1, y1

            # draw label and score
            if label and label_x is not None and label_y is not None:
                text = ann.label
                if score and ann.score is not None:
                    text = f"{ann.label} ({ann.score:.2f})"
                draw_ctx.text((label_x + int(outline_width * 1.5), label_y), text, font=font, fill=col)
        
        # overlay mask on original image
        if 'overlay' in layers:
            if mask_image is not None and mask_image.any():
                mask_pil = PIL.Image.fromarray(mask_image.astype(np.uint8))
                im = PIL.Image.blend(im, mask_pil, alpha=0.5)
        
        if output is not None:
            im.save(output)

        return im
    



@dataclass
class ImageDataset:
    """Represents a dataset of images with their corresponding annotations.
    
    Manages multiple ImageRecord objects (images with annotations). Supports creating
    datasets from COCO format, serialization to dictionary and COCO formats, and dataset operations.
    
    Args:
        records (list[ImageRecord]): List of image records. Default empty list.
    """
    records: list[ImageRecord] = field(default_factory=list)

    @classmethod
    def from_coco(
        cls,
        coco_dict: dict[str, typing.Any],
        image_root: str | os.PathLike[str] | None = None,
    ) -> ImageDataset:
        """Create an ImageDataset from a COCO-style dictionary.
        
        Args:
            coco_dict (dict): COCO format dictionary with 'images', 'annotations', 'categories' keys.
            image_root (str | os.PathLike | None): Root directory for image paths. Default None.
            
        Returns:
            ImageDataset: Dataset initialized from COCO dictionary.
            
        Examples:
            >>> coco_data = json.load(open('annotations.json'))
            >>> dataset = ImageDataset.from_coco(coco_data, image_root='images/')
        """
        images = coco_dict.get("images", [])
        annotations = coco_dict.get("annotations", [])
        categories = coco_dict.get("categories", [])

        cateid2name = {cat["id"]: cat["name"] for cat in categories}
        
        imid2record: dict[int, ImageRecord] = {}
        for img in images:
            image_id = img["id"]
            source = img["file_name"]
            if image_root is not None:
                source = os.path.join(image_root, source)
            imid2record[image_id] = ImageRecord(source=source)

        for ann in annotations:
            image_id = ann["image_id"]
            label_id = ann["category_id"]
            label_name = cateid2name[label_id]
                        
            bbox = None
            if 'bbox' in ann and ann['bbox'] is not None:
                bbox = Bbox.from_xywh(*ann['bbox'], imsize=imid2record[image_id].size)

            segm = None
            segm_data = ann.get("segmentation")
            if segm_data is not None:
                segm_format = "rle" if isinstance(segm_data, dict) else "polygon"
                if segm_format == 'rle':
                    segm = Segm.from_rle(segm_data, imsize=imid2record[image_id].size)
                elif len(segm_data) > 0:
                    segm = Segm.from_polygons(segm_data, imsize=imid2record[image_id].size)
                            
            score = ann.get("score")

            imid2record[image_id].add_annotation(
                InstanceAnnotation(label=label_name, bbox=bbox, segm=segm, score=score)
            )
            
        dataset_records = list(imid2record.values())
        return cls(records=dataset_records)


    def __iter__(self):
        """Make ImageDataset iterable over its records.
        
        Yields:
            ImageRecord: Individual image records in the dataset.
        """
        return iter(self.records)


    @property
    def size(self) -> int:
        """Return the number of image records in the dataset.
        
        Returns:
            int: Number of images.
        """
        return len(self.records)
    

    def append(self, image: ImageRecord) -> None:
        """Append an ImageRecord to the dataset after validation.
        
        Args:
            image (ImageRecord): Image record to append.
            
        Raises:
            TypeError: If image is not ImageRecord.
        """
        if not isinstance(image, ImageRecord):
            raise TypeError("image must be an ImageRecord.")
        self.records.append(image)


    def to_dict(self) -> dict[str, typing.Any]:
        """Convert ImageDataset to a dictionary representation.
        
        Returns:
            dict: Dictionary with 'records' key containing list of image dictionaries.
            
        Examples:
            >>> dataset = ImageDataset(records=[...])
            >>> d = dataset.to_dict()
        """
        return {
            "records": [
                record.to_dict() for record in self.records
            ],
        }
    
    
    def to_coco(
        self,
        image_root: str | os.PathLike[str] | None = None
    ) -> dict[str, typing.Any]:
        """Convert ImageDataset to a COCO-style dictionary.
        
        Args:
            image_root (str | os.PathLike | None): Root directory for relative image paths. Default None.
            
        Returns:
            dict: COCO format dictionary with 'images', 'annotations', 'categories' keys.
            
        Examples:
            >>> dataset = ImageDataset.from_coco(coco_dict)
            >>> coco_output = dataset.to_coco()
            >>> with open('output.json', 'w') as f:
            ...     json.dump(coco_output, f)
        """
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        category_id_map = {}
        annotation_id = 1

        for image_id, image in enumerate(self.records, start=1):
            if image_root is not None:
                file_name = str(pathlib.Path(image.source).relative_to(image_root))
            else:
                file_name = str(image.source)
                
            coco_dict["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": image.size[0],
                "height": image.size[1],
            })

            for ann in image.annotations:
                if ann.label not in category_id_map:
                    category_id = len(category_id_map) + 1
                    category_id_map[ann.label] = category_id
                    coco_dict["categories"].append({
                        "id": category_id,
                        "name": ann.label,
                    })
                else:
                    category_id = category_id_map[ann.label]
                
                coco_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": ann.bbox.to_xywh() if ann.bbox else None,
                    "segmentation": ann.segm.to_rle() if ann.segm is not None else None,
                    "area": ann.area,
                    "iscrowd": 0,
                })
                annotation_id += 1

        return coco_dict

