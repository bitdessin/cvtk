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
        """Create bbox from top-left corner (x, y) and width, height. imsize is required."""
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
        """Create bbox from top-left (x1, y1) and bottom-right (x2, y2) corners. imsize is required."""
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
        """Create bbox from center (cx, cy) and width, height. imsize is required."""
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
        """Create bbox from normalized coordinates [0, 1]. imsize is required."""
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
        """Create bbox from normalized coordinates [0, 1]. imsize is required."""
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
        """Create bbox from normalized center and dimensions [0, 1]. imsize is required."""
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
        return self.x_max - self.x_min


    @property
    def height(self) -> float:
        return self.y_max - self.y_min


    @property
    def area(self) -> float:
        return float(self.width * self.height)


    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)


    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, w, h) format."""
        return (self.x_min, self.y_min, self.width, self.height)
    
    
    def to_cxcywh(self) -> tuple[float, float, float, float]:
        """Convert to (cx, cy, w, h) format."""
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        return (cx, cy, self.width, self.height)

    
    def to_xyxyr(self) -> tuple[float, float, float, float]:
        """Convert to normalized (x1, y1, x2, y2) format in range [0, 1]."""
        x1r = self.x_min / self.imsize[0]
        y1r = self.y_min / self.imsize[1]
        x2r = self.x_max / self.imsize[0]
        y2r = self.y_max / self.imsize[1]
        return (x1r, y1r, x2r, y2r)

    
    def to_xywhr(self) -> tuple[float, float, float, float]:
        """Convert to normalized (x, y, w, h) format in range [0, 1]."""
        xr = self.x_min / self.imsize[0]
        yr = self.y_min / self.imsize[1]
        wr = self.width / self.imsize[0]
        hr = self.height / self.imsize[1]
        return (xr, yr, wr, hr)
    
    
    def to_cxcywhr(self) -> tuple[float, float, float, float]:
        """Convert to normalized (cx, cy, w, h) format in range [0, 1]."""
        cxr = (self.x_min + self.x_max) / 2.0 / self.imsize[0]
        cyr = (self.y_min + self.y_max) / 2.0 / self.imsize[1]
        wr = self.width / self.imsize[0]
        hr = self.height / self.imsize[1]
        return (cxr, cyr, wr, hr)
    
    
    @staticmethod
    def xyxy2xywh(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
        x1, y1, x2, y2 = xyxy
        return (x1, y1, x2 - x1, y2 - y1)
    
    
    @staticmethod
    def xywh2xyxy(xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
        x, y, w, h = xywh
        return (x, y, x + w, y + h)

    
    @staticmethod
    def xywh2cxcywh(xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x, y, w, h) to (cx, cy, w, h)."""
        x, y, w, h = xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        return (cx, cy, w, h)

    
    @staticmethod
    def cxcywh2xywh(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (cx, cy, w, h) to (x, y, w, h)."""
        cx, cy, w, h = cxcywh
        x = cx - w / 2.0
        y = cy - h / 2.0
        return (x, y, w, h)

    
    @staticmethod
    def xyxy2cxcywh(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)

    
    @staticmethod
    def cxcywh2xyxy(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
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
        """Convert normalized (x1, y1, x2, y2) to normalized (x, y, w, h)."""
        x1r, y1r, x2r, y2r = xyxyr
        return (x1r, y1r, x2r - x1r, y2r - y1r)
    
    
    @staticmethod
    def xywhr2xyxyr(xywhr: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert normalized (x, y, w, h) to normalized (x1, y1, x2, y2)."""
        xr, yr, wr, hr = xywhr
        return (xr, yr, xr + wr, yr + hr)


@dataclass
class Segm:
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
        mask_array = np.asarray(mask, dtype=bool)
        return cls(_mask=mask_array, _imsize=(mask_array.shape[1], mask_array.shape[0]))


    @classmethod
    def from_rle(
        cls,
        rle: _RLE,
        *,
        imsize: _ImgSize | None = None,
    ) -> Segm:
        rel_size = rle.get("size")
        if rel_size is not None:
            # RLE size is [height, width], but imsize should be (width, height)
            imsize = imsize or (rel_size[1], rel_size[0])

        if imsize is None:
            raise ValueError("imsize is required when rle has no size.")

        return cls(_rle=dict(rle), _imsize=imsize)


    @classmethod
    def from_polygons(
        cls,
        polygons: _Polygons,
        *,
        imsize: _ImgSize | None = None,
    ) -> Segm:
        return cls(_polygons=[list(poly) for poly in polygons], _imsize=imsize)


    @property
    def size(self) -> _ImgSize:
        return self._imsize
    
    
    @property
    def area(self) -> float:
        return float(np.sum(self.to_mask()))


    def to_mask(
        self,
    ) -> _MaskArray:
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
        if self._rle is None:
            self._rle = self._mask_to_rle(self.to_mask(), compressed=compressed)
        return copy.deepcopy(self._rle)


    def to_polygons(
        self
    ) -> _Polygons:
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



    def to_dict(self, *, format: str = "rle") -> dict[str, typing.Any]:
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
    label: str
    bbox: Bbox | None = None
    segm: Segm | None = None
    score: float | None = None
 
    @property
    def area(self) -> float | None:
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
    source: str | os.PathLike[str]
    annotations: list[InstanceAnnotation] = field(default_factory=list)


    def __post_init__(self) -> None:
        im = cvtk.io.imread(self.source)

        self.source = pathlib.Path(self.source)
        self.size = im.size
        self.annotations = list(self.annotations)

        self._validate_annotations()

    def add_annotation(self, annotation: InstanceAnnotation) -> None:
        if not isinstance(annotation, InstanceAnnotation):
            raise TypeError("annotation must be an InstanceAnnotation.")

        self._validate_annotation(annotation)
        self.annotations.append(annotation)


    def _validate_annotations(self) -> None:
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
        """Draw annotations on an image
        
        Args:
            layers: str | list[str]: The layers to draw. Can be a single layer or a list of layers. 
                Valid layers: 'bbox' (bounding boxes), 'segm' (segmentation polygons), 
                'mask' (colored mask with black background), 'overlay' (mask blended on original image).
                Default is 'bbox'.
            output: str | os.PathLike[str] | None: The output file path to save the drawn image. 
                If None, the image will not be saved. Default is None.
            cutoff: float: The score cutoff for drawing annotations. Annotations with scores below 
                this value will not be drawn. Default is 0.0.
            label: bool: Whether to draw the label of the annotation. Default is True.
            score: bool: Whether to draw the score of the annotation. Default is True.
            font: PIL.ImageFont.ImageFont | None: The font to use for drawing the label and score. 
                If None, a default font will be used. Default is None.
            colors: dict[str, tuple] | None: A dictionary mapping label names to RGB colors. 
                If None, random colors will be generated. Default is None.
            
        Returns:
            PIL.Image.Image: The drawn image.
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
    records: list[ImageRecord] = field(default_factory=list)

    @classmethod
    def from_coco(
        cls,
        coco_dict: dict[str, typing.Any],
        image_root: str | os.PathLike[str] | None = None,
    ) -> ImageDataset:
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


    @property
    def size(self) -> int:
        return len(self.records)
    

    def append(self, image: ImageRecord) -> None:
        if not isinstance(image, ImageRecord):
            raise TypeError("image must be an ImageRecord.")
        self.records.append(image)


    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "records": [
                record.to_dict() for record in self.records
            ],
        }
    
    
    def to_coco(
        self,
        image_root: str | os.PathLike[str] | None = None
    ) -> dict[str, typing.Any]:
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

