cvtk.data
#########

This module provides classes and functions for working with image records, annotations,
and datasets. It supports multiple annotation formats including bounding boxes, segmentation masks,
and COCO format.


Main Classes
************

- **Bbox**: Bounding box representation with multiple coordinate formats (xyxy, xywh, cxcywh, etc.)
- **Segm**: Segmentation representation supporting masks, RLE, and polygon formats
- **InstanceAnnotation**: Combined annotation with label, bounding box, segmentation, and score
- **ImageRecord**: Single image with its annotations
- **ImageDataset**: Collection of image records with dataset-level operations


API Reference
*************

.. automodule:: cvtk.data
    :noindex:
    :members:
    :no-undoc-members:
    :member-order: bysource


.. automodule:: cvtk.data.coco
    :noindex:
    :members:
    :no-undoc-members:
    :member-order: bysource


