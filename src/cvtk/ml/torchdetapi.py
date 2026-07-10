from __future__ import annotations
import os
import json
import gzip
import pathlib
import gc
import warnings
import inspect
from typing import Any
import numpy as np
import filetype
import PIL.Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision
import torchvision.transforms.v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from .. import data as cvtk_data
from .. import utils as cvtk_utils
from . import data as cvtk_ml_data


def _resolve_image_path(img_path: str | None, image_base: str) -> str | None:
    if img_path is None or img_path == '':
        return img_path
    if os.path.isabs(img_path):
        return img_path
    return os.path.join(image_base, img_path)


class DataTransform():
    """Image preprocessing pipeline for torchvision detection models.

    This initial implementation keeps transforms simple and geometry-safe for
    detection targets. It converts images into float tensors in [0, 1].
    """
    def __init__(self, is_train: bool = False) -> None:
        """Create the transform pipeline.

        Args:
            is_train (bool): Flag indicating whether the transform will be used for
                training. The current pipeline is the same for train and eval,
                but the flag is kept for future augmentation support.
        """
        self.is_train = is_train
        self.pipeline = torchvision.transforms.Compose([
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ])


class Dataset(torch.utils.data.Dataset):
    """Detection dataset supporting COCO annotations and raw image inputs.

    Args:
        datalabel: DataLabel-like object, label file, or list of class names.
        dataset: COCO json path, image directory, image file path, or list of image paths.
        transform: DataTransform or torchvision compose transform.
        image_root: Base directory for COCO image paths.
    """
    def __init__(
        self,
        datalabel: cvtk_ml_data.DataLabel | str | list[str] | tuple[str],
        dataset: str | list[str] | tuple[str],
        transform: DataTransform | torchvision.transforms.Compose | None = None,
        image_root: str | None = None
    ) -> None:
        """Create a detection dataset.

        Args:
            datalabel (cvtk_ml_data.DataLabel|str|list|tuple): A DataLabel,
                label file, or list of class names.
            dataset (str|list|tuple): COCO annotation file, image file, image
                directory, text file of image paths, or a list of image paths.
            transform (cvtk.ml.torchdetapi.DataTransform|torchvision.transforms.Compose|None):
                Optional transform applied to each image.
            image_root (str|None): Optional base directory used to resolve
                relative COCO image paths.

        Raises:
            ValueError: If no images can be loaded from ``dataset``.
            TypeError: If ``datalabel`` or ``dataset`` has an unsupported type.
        """
        if isinstance(datalabel, cvtk_ml_data.DataLabel):
            self.datalabel = datalabel
        else:
            self.datalabel = cvtk_ml_data.DataLabel(datalabel)

        if isinstance(transform, DataTransform):
            transform = transform.pipeline
        if transform is None:
            transform = DataTransform(is_train=False).pipeline
        self.transform = transform

        self.image_root = image_root
        self.ann_file = None
        self.samples = []
        self.has_targets = False
        self._load(dataset)

        if len(self.samples) == 0:
            raise ValueError('No images are loaded. Check the dataset.')


    def _load(self, dataset: str | list[str] | tuple[str]) -> None:
        if isinstance(dataset, str) and os.path.isfile(dataset) and dataset.endswith(('.json', '.json.gz', '.json.gzip')):
            self._load_from_coco(dataset)
            return

        if isinstance(dataset, str):
            if os.path.isfile(dataset):
                if filetype.is_image(dataset):
                    self.samples = [{'path': dataset, 'target': None}]
                    return

                lines = []
                if dataset.endswith(('.gz', '.gzip')):
                    fh = gzip.open(dataset, 'rt')
                else:
                    fh = open(dataset, 'r')
                with fh:
                    for line in fh:
                        words = line.rstrip().split('\t')
                        if len(words) == 0 or words[0] == '':
                            continue
                        lines.append(words[0])
                self.samples = [{'path': p, 'target': None} for p in lines]
                return

            if os.path.isdir(dataset):
                images = []
                for root, dirs, files in os.walk(dataset):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        if filetype.is_image(fpath):
                            images.append(fpath)
                self.samples = [{'path': p, 'target': None} for p in images]
                return

        if isinstance(dataset, (list, tuple)):
            samples = []
            for item in dataset:
                if isinstance(item, (list, tuple)):
                    samples.append({'path': item[0], 'target': None})
                else:
                    samples.append({'path': item, 'target': None})
            self.samples = samples
            return

        raise TypeError(f'Invalid dataset type: {type(dataset)}')


    def _load_from_coco(self, ann_file: str) -> None:
        self.ann_file = os.path.abspath(ann_file)

        if ann_file.endswith(('.gz', '.gzip')):
            with gzip.open(ann_file, 'rt') as fh:
                coco_dict = json.load(fh)
        else:
            with open(ann_file, 'r') as fh:
                coco_dict = json.load(fh)

        if 'images' not in coco_dict or 'annotations' not in coco_dict or 'categories' not in coco_dict:
            raise ValueError(f'Invalid COCO format: {ann_file}')

        coco_categories = sorted(coco_dict['categories'], key=lambda x: x['id'])
        coco_names = [cat['name'] for cat in coco_categories]
        if coco_names != list(self.datalabel.labels):
            raise ValueError(
                'Class names are different between annotations and datalabel. '
                f'annotation={coco_names}, datalabel={self.datalabel.labels}'
            )

        image_base = os.path.dirname(os.path.abspath(ann_file)) if self.image_root is None else os.path.abspath(self.image_root)
        image_id_by_path = {}
        for im in coco_dict['images']:
            resolved_path = _resolve_image_path(im.get('file_name'), image_base)
            image_id_by_path[os.path.abspath(resolved_path)] = im['id']

        image_dataset = cvtk_data.ImageDataset.from_coco(coco_dict, image_root=image_base)

        self.samples = []
        for record in image_dataset.records:
            boxes = []
            labels = []
            area = []
            iscrowd = []
            masks = []

            for ann in record.annotations:
                if ann.bbox is None:
                    continue
                x1, y1, x2, y2 = ann.bbox.to_xyxy()
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x1, y1, x2, y2])
                # torchvision detection labels are 1..N, where 0 is background.
                labels.append(self.datalabel[ann.label] + 1)
                area.append(float(ann.area) if ann.area is not None else float(w * h))
                iscrowd.append(0)
                masks.append(ann.segm.to_mask().astype(np.uint8) if ann.segm is not None else None)

            record_path = os.path.abspath(str(record.source))
            if record_path not in image_id_by_path:
                raise ValueError(f'Image path from COCO record not found in image table: {record_path}')

            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id_by_path[record_path],
                'area': area,
                'iscrowd': iscrowd,
                'masks': masks,
            }

            self.samples.append({
                'path': str(record.source),
                'target': target,
            })

        self.has_targets = True


    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, str]:
        sample = self.samples[i]
        img_path = sample['path']
        target = sample['target']

        img = PIL.Image.open(img_path).convert('RGB')
        width, height = img.size

        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            return img, None, img_path

        boxes = torch.tensor(target['boxes'], dtype=torch.float32)
        labels = torch.tensor(target['labels'], dtype=torch.int64)
        area = torch.tensor(target['area'], dtype=torch.float32)
        iscrowd = torch.tensor(target['iscrowd'], dtype=torch.int64)
        masks = target.get('masks', None)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        target_dict = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([target['image_id']], dtype=torch.int64),
            'area': area,
            'iscrowd': iscrowd,
        }

        if masks is not None and len(masks) == len(target['boxes']) and len(masks) > 0:
            if any(m is not None for m in masks):
                masks_ = []
                for m in masks:
                    if m is None:
                        m = np.zeros((height, width), dtype=np.uint8)
                    masks_.append(m)
                target_dict['masks'] = torch.as_tensor(np.stack(masks_, axis=0), dtype=torch.uint8)

        return img, target_dict, img_path


    def __len__(self) -> int:
        return len(self.samples)



class DataLoader(torch.utils.data.DataLoader):
    """DataLoader wrapper for detection tasks with default collate function."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a detection dataloader.

        Args:
            *args: Positional arguments forwarded to
                :class:`torch.utils.data.DataLoader`.
            **kwargs: Keyword arguments forwarded to
                :class:`torch.utils.data.DataLoader`.

        Notes:
            If no ``collate_fn`` is provided, this injects the detection-aware
            collate function that keeps images, targets, and paths grouped
            separately.
        """
        if 'collate_fn' not in kwargs or kwargs['collate_fn'] is None:
            kwargs['collate_fn'] = _detection_collate_fn
        super().__init__(*args, **kwargs)


def _detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor] | None, str]]
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]] | None, list[str]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    if all(t is None for t in targets):
        targets = None
    return images, targets, paths



class DetRunner():
    """Torchvision detection runner.

    Initial implementation supports Faster R-CNN and is intentionally structured
    to be extendable to other torchvision detection architectures later.
    """
    def __init__(
        self,
        datalabel: cvtk_ml_data.DataLabel | str | list[str] | tuple[str],
        model: str | torch.nn.Module = 'fasterrcnn_resnet50_fpn',
        weights: str | Any | None = None,
        workspace: str | None = None,
        device: str = 'auto'
    ) -> None:
        """Create a torchvision detection runner.

        Args:
            datalabel (cvtk.ml.data.DataLabel|str|list|tuple): A DataLabel,
                label file, or list of class names.
            model (str|torch.nn.Module): Torchvision detection model name or
                an instantiated module.
            weights (str|torchvision.models.detection weights|None): Optional
                weights enum name, checkpoint path, or weights object accepted
                by torchvision.
            workspace (str|None): Optional output directory for checkpoints and
                reports.
            device (str): Target device name or ``'auto'`` to pick CUDA when
                available.
        """
        self.task_type = 'det'
        self.device = self._init_device(device)
        self.datalabel = self._init_datalabel(datalabel)
        self.workspace = self._init_workspace(workspace)
        self.train_stats = {'epoch': [], 'train_loss': [], 'valid_loss': []}
        self.test_stats = None
        self.model_name = model
        self.model = self._build_model(model, weights, len(self.datalabel.labels) + 1).to(self.device)


    def __del__(self):
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
        except:
            pass


    def _init_device(self, device: str = 'auto') -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device


    def _init_datalabel(
        self,
        datalabel: cvtk_ml_data.DataLabel | str | list[str] | tuple[str]
    ) -> cvtk_ml_data.DataLabel:
        if isinstance(datalabel, cvtk_ml_data.DataLabel):
            return datalabel
        if isinstance(datalabel, (str, list, tuple)):
            return cvtk_ml_data.DataLabel(datalabel)
        raise TypeError(f'Invalid datalabel type: {type(datalabel)}')


    def _init_workspace(self, workspace: str | None) -> str | None:
        if workspace is not None and not os.path.exists(workspace):
            os.makedirs(workspace)
        return workspace


    def _resolve_torchvision_det_attr(self, attr_name: str) -> Any:
        obj = torchvision.models.detection
        for attr in attr_name.split('.'):
            obj = getattr(obj, attr)
        return obj


    def _resolve_default_torchvision_weights(self, model_builder: Any) -> Any | None:
        """Resolve DEFAULT torchvision weights enum for a model builder.

        Returns:
            The DEFAULT weights enum value when available, otherwise None.
        """
        try:
            enum_cls = torchvision.models.get_model_weights(model_builder)
            return enum_cls.DEFAULT
        except Exception:
            return None


    def _replace_detection_head(self, model: torch.nn.Module, arch_name: str, num_classes: int) -> torch.nn.Module:
        # Faster R-CNN family
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'box_predictor'):
            if not hasattr(model, 'roi_heads') or not hasattr(model.roi_heads, 'box_predictor'):
                raise ValueError(f'Unsupported Faster R-CNN model structure: {arch_name}')
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model

        # RetinaNet family
        if hasattr(model, 'head') and hasattr(model.head, 'classification_head'):
            cls_head = model.head.classification_head
            if isinstance(cls_head, RetinaNetClassificationHead) or str(arch_name).startswith('retinanet'):
                if not hasattr(cls_head, 'cls_logits') or not hasattr(cls_head, 'num_anchors'):
                    raise ValueError(f'Unsupported RetinaNet head structure: {type(cls_head)}')
                in_channels = cls_head.cls_logits.in_channels
                num_anchors = cls_head.num_anchors
                model.head.classification_head = RetinaNetClassificationHead(
                    in_channels=in_channels,
                    num_anchors=num_anchors,
                    num_classes=num_classes,
                )
                return model

            # SSDLite family
            if isinstance(cls_head, SSDLiteClassificationHead) or str(arch_name).startswith('ssdlite'):
                if not hasattr(model, 'anchor_generator'):
                    raise ValueError('SSDLite model missing anchor_generator.')
                num_anchors = model.anchor_generator.num_anchors_per_location()
                # Each module ends with pointwise conv at index [1].
                in_channels = [layer[1].in_channels for layer in cls_head.module_list]

                norm_layer = torch.nn.BatchNorm2d
                try:
                    bn_ref = cls_head.module_list[0][0][1]
                    if isinstance(bn_ref, torch.nn.BatchNorm2d):
                        def _norm_layer(c):
                            return torch.nn.BatchNorm2d(c, eps=bn_ref.eps, momentum=bn_ref.momentum)
                        norm_layer = _norm_layer
                except Exception:
                    pass

                model.head.classification_head = SSDLiteClassificationHead(
                    in_channels=in_channels,
                    num_anchors=num_anchors,
                    num_classes=num_classes,
                    norm_layer=norm_layer,
                )
                return model

            # SSD family
            if isinstance(cls_head, SSDClassificationHead) or str(arch_name).startswith('ssd'):
                if not hasattr(model, 'anchor_generator'):
                    raise ValueError('SSD model missing anchor_generator.')
                num_anchors = model.anchor_generator.num_anchors_per_location()
                in_channels = [layer.in_channels for layer in cls_head.module_list]
                model.head.classification_head = SSDClassificationHead(
                    in_channels=in_channels,
                    num_anchors=num_anchors,
                    num_classes=num_classes,
                )
                return model

        raise ValueError(
            f'Unsupported detection architecture: {arch_name}. '
            'Current implementation supports Faster R-CNN, RetinaNet, SSD, and SSDLite.'
        )


    def _build_model(
        self,
        model: str | torch.nn.Module,
        weights: str | Any | None,
        num_classes: int
    ) -> torch.nn.Module:
        if isinstance(model, torch.nn.Module):
            module = self._replace_detection_head(model, model.__class__.__name__.lower(), num_classes)
            if isinstance(weights, str) and os.path.exists(weights):
                state_dict = torch.load(weights, map_location='cpu')
                module.load_state_dict(state_dict, strict=False)
            return module

        if not isinstance(model, str):
            raise TypeError(f'Invalid model type: {type(model)}')

        try:
            model_builder = self._resolve_torchvision_det_attr(model)
        except AttributeError as exc:
            raise ValueError(f'Unknown torchvision detection model: {model}') from exc

        if not callable(model_builder):
            raise ValueError(f'Invalid torchvision detection model: {model}')

        builder_sig = inspect.signature(model_builder)
        builder_kwargs = {}
        # Avoid unnecessary backbone download when loading local checkpoint.
        if 'weights_backbone' in builder_sig.parameters and isinstance(weights, str) and os.path.exists(weights):
            builder_kwargs['weights_backbone'] = None

        if isinstance(weights, str) and os.path.exists(weights):
            module = model_builder(weights=None, **builder_kwargs)
        elif weights is None:
            default_weights = self._resolve_default_torchvision_weights(model_builder)
            if default_weights is not None:
                module = model_builder(weights=default_weights, **builder_kwargs)
            else:
                module = model_builder(weights=None, **builder_kwargs)
        elif isinstance(weights, str):
            try:
                resolved_weights = self._resolve_torchvision_det_attr(weights)
            except AttributeError as exc:
                raise ValueError(
                    f'Unknown torchvision detection weights: {weights}. '
                    'Use a valid enum path like FasterRCNN_ResNet50_FPN_Weights.DEFAULT '
                    'or provide a local checkpoint path.'
                ) from exc
            module = model_builder(weights=resolved_weights)
        else:
            module = model_builder(weights=weights)

        module = self._replace_detection_head(module, model, num_classes)

        if isinstance(weights, str) and os.path.exists(weights):
            state_dict = torch.load(weights, map_location='cpu')
            module.load_state_dict(state_dict, strict=False)

        return module


    def _to_device_targets(self, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        targets_device = []
        for target in targets:
            target_device = {}
            for key, value in target.items():
                if torch.is_tensor(value):
                    target_device[key] = value.to(self.device)
                else:
                    target_device[key] = value
            targets_device.append(target_device)
        return targets_device


    def _train_or_eval_loss(
        self,
        dataloader: torch.utils.data.DataLoader | None,
        optimizer: torch.optim.Optimizer | None = None
    ) -> float | None:
        if dataloader is None:
            return None

        # Torchvision detection models return loss dict only in train mode.
        # For validation, keep train mode but disable gradients via set_grad_enabled.
        self.model.train()

        running_loss = 0.0
        n_batches = 0

        for images, targets, paths in dataloader:
            if targets is None:
                raise ValueError('Training/validation requires labeled COCO dataset targets.')

            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = self._to_device_targets(targets)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(optimizer is not None):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if optimizer is not None:
                losses.backward()
                optimizer.step()

            running_loss += losses.detach().item()
            n_batches += 1

        if n_batches == 0:
            raise ValueError('No samples were processed from dataloader.')

        return running_loss / n_batches


    def _predict_as_imagedataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        cutoff: float = 0.5
    ) -> cvtk_data.ImageDataset:
        self.model.eval()
        records = []

        with torch.no_grad():
            for images, targets, paths in dataloader:
                images_device = [img.to(self.device, non_blocking=True) for img in images]
                outputs = self.model(images_device)

                for image_tensor, img_path, pred in zip(images, paths, outputs):
                    annotations = []
                    h = int(image_tensor.shape[-2])
                    w = int(image_tensor.shape[-1])
                    imsize = (w, h)

                    boxes = pred.get('boxes', torch.empty((0, 4)))
                    labels = pred.get('labels', torch.empty((0,), dtype=torch.int64))
                    scores = pred.get('scores', torch.empty((0,), dtype=torch.float32))

                    boxes = boxes.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    scores = scores.detach().cpu().numpy()

                    for box, label_id, score in zip(boxes, labels, scores):
                        if float(score) < cutoff:
                            continue
                        label_index = int(label_id) - 1
                        if label_index < 0 or label_index >= len(self.datalabel.labels):
                            continue

                        bbox = cvtk_data.Bbox.from_xyxy(*box.tolist(), imsize=imsize)
                        ann = cvtk_data.InstanceAnnotation(
                            label=self.datalabel[label_index],
                            bbox=bbox,
                            segm=None,
                            score=float(score),
                        )
                        annotations.append(ann)

                    records.append(
                        cvtk_data.ImageRecord(
                            source=pathlib.Path(img_path),
                            annotations=annotations,
                            size=imsize,
                        )
                    )

        return cvtk_data.ImageDataset(records=records)


    def train(
        self,
        train: torch.utils.data.DataLoader,
        valid: torch.utils.data.DataLoader | None = None,
        test: torch.utils.data.DataLoader | None = None,
        epoch: int = 20,
        optimizer: torch.optim.Optimizer | str = 'auto',
        scheduler: Any | None = None
    ) -> None:
        """Train the detection model.

        Args:
            train (torch.utils.data.DataLoader): Training dataloader built from
                labeled COCO targets.
            valid (torch.utils.data.DataLoader|None): Optional validation
                dataloader.
            test (torch.utils.data.DataLoader|None): Optional test dataloader
                evaluated after training completes.
            epoch (int): Number of epochs to train.
            optimizer (torch.optim.Optimizer|str): Optimizer instance or
                ``'auto'`` to use SGD.
            scheduler (torch.optim.lr_scheduler._LRScheduler|None): Optional
                learning-rate scheduler stepped once per epoch.

        Raises:
            ValueError: If a training or validation batch has no targets.
        """
        if optimizer == 'auto':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=1e-3,
                momentum=0.9,
                weight_decay=5e-4,
            )

        for epoch_i in range(1, epoch + 1):
            train_loss = self._train_or_eval_loss(train, optimizer=optimizer)
            valid_loss = self._train_or_eval_loss(valid, optimizer=None) if valid is not None else None

            self.train_stats['epoch'].append(epoch_i)
            self.train_stats['train_loss'].append(train_loss)
            self.train_stats['valid_loss'].append(valid_loss)

            print(f'Epoch {epoch_i}/{epoch} train_loss={train_loss:.4f} valid_loss={valid_loss}')

            if scheduler is not None:
                scheduler.step()

            if self.workspace is not None:
                self.save(os.path.join(self.workspace, 'checkpoint_latest.pth'))

        if test is not None:
            self.test(test)


    def test(
        self,
        dataloader: torch.utils.data.DataLoader,
        cutoff: float = 0.5
    ) -> dict[str, Any]:
        """Evaluate predictions against COCO ground truth.

        Args:
            dataloader (torch.utils.data.DataLoader): Detection dataloader
                built from a COCO dataset.
            cutoff (float): Minimum score threshold for predicted annotations.

        Returns:
            dict: The computed COCO metrics dictionary.

        Raises:
            TypeError: If ``dataloader`` is not a torch dataloader.
            ValueError: If the dataloader dataset does not expose a COCO
                annotation file.
        """
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError(f'`dataloader` must be torch.utils.data.DataLoader, but got {type(dataloader)}.')

        dataset = getattr(dataloader, 'dataset', None)
        gt_ann_file = getattr(dataset, 'ann_file', None)
        if gt_ann_file is None:
            raise ValueError('`test` requires a COCO dataset input. Provide Dataset built from COCO annotation file.')

        pred_dataset = self._predict_as_imagedataset(dataloader, cutoff=cutoff)

        # Build COCO output from ImageDataset and then add score fields required by COCOeval.
        pred_coco = pred_dataset.to_coco()

        # Align prediction image ids/file_names with GT COCO entries to avoid mapping mismatch.
        image_id_map = {}
        if hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            if len(dataset.samples) != len(pred_coco['images']):
                raise ValueError(
                    'Prediction image count mismatch: '
                    f'pred={len(pred_coco["images"])}, dataset={len(dataset.samples)}'
                )

            gt_base = os.path.dirname(os.path.abspath(gt_ann_file))
            for pred_image, sample in zip(pred_coco['images'], dataset.samples):
                old_id = pred_image['id']
                new_id = sample['target']['image_id'] if sample.get('target') is not None else old_id
                image_id_map[old_id] = new_id
                pred_image['id'] = new_id

                img_path = os.path.abspath(sample['path'])
                try:
                    pred_image['file_name'] = str(pathlib.Path(img_path).relative_to(gt_base))
                except ValueError:
                    pred_image['file_name'] = img_path

            for ann_json in pred_coco['annotations']:
                ann_json['image_id'] = image_id_map.get(ann_json['image_id'], ann_json['image_id'])

        # Keep category table aligned with datalabel even if some classes are not predicted.
        pred_coco['categories'] = [
            {'id': i + 1, 'name': label}
            for i, label in enumerate(self.datalabel.labels)
        ]

        ann_scores = []
        for record in pred_dataset.records:
            for ann in record.annotations:
                if ann.bbox is None:
                    continue
                ann_scores.append(float(ann.score) if ann.score is not None else 0.0)

        if len(ann_scores) != len(pred_coco['annotations']):
            raise ValueError('Prediction COCO annotation count mismatch while attaching scores.')

        for ann_json, score in zip(pred_coco['annotations'], ann_scores):
            ann_json['score'] = score

        if len(pred_coco['annotations']) == 0:
            warnings.warn('No predictions were produced for test dataset. COCO metrics may be undefined.')

        output_dir = self.workspace if self.workspace is not None else os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        pred_coco_fpath = os.path.join(output_dir, 'test_outputs.coco.json')
        cvtk_utils.save_json(pred_coco, pred_coco_fpath, indent=4, ensure_ascii=False)

        if len(pred_coco['annotations']) == 0:
            self.test_stats = {
                'stats': {},
                'class_stats': {},
            }
            return self.test_stats

        self.test_stats = cvtk_data.coco.calc_stats(
            gt_ann_file,
            pred_coco_fpath,
            image_by='id',
            category_by='name',
            iouType='bbox' if self.task_type == 'det' else 'segm',
        )

        return self.test_stats


    def save(
        self,
        output: str
    ) -> None:
        """Save the model, label map, and training statistics.

        Args:
            output (str): Output path for the model checkpoint. ``.pth`` is
                appended automatically when missing.
        """
        if not output.endswith('.pth'):
            output += '.pth'

        outdir = os.path.dirname(output)
        if outdir != '' and not os.path.exists(outdir):
            os.makedirs(outdir)

        self.model = self.model.to('cpu')
        torch.save(self.model.state_dict(), output)
        self.model = self.model.to(self.device)

        self.datalabel.save(os.path.splitext(output)[0] + '.dl.txt')

        train_stats_path = os.path.splitext(output)[0] + '.train_stats.txt'
        with open(train_stats_path, 'w') as fh:
            fh.write('\t'.join(self.train_stats.keys()) + '\n')
            for vals in zip(*self.train_stats.values()):
                fh.write('\t'.join(['NA' if v is None else str(v) for v in vals]) + '\n')

        if self.test_stats is not None:
            test_stats_path = os.path.splitext(output)[0] + '.test_stats.json'
            with open(test_stats_path, 'w') as fh:
                json.dump(self.test_stats, fh, indent=4, ensure_ascii=False)


    def inference(
        self,
        data: torch.utils.data.DataLoader | str | list[str] | tuple[str],
        cutoff: float = 0.5,
        batch_size: int = 4,
        num_workers: int = 0
    ) -> cvtk_data.ImageDataset:
        """Run inference and return an :class:`cvtk.data.ImageDataset`.

        Args:
            data (torch.utils.data.DataLoader|str|list|tuple): Existing
                dataloader or an image path, directory, list of image paths,
                or text file accepted by :class:`Dataset`.
            cutoff (float): Minimum score threshold for returned predictions.
            batch_size (int): Batch size used when ``data`` is not already a
                dataloader.
            num_workers (int): Number of worker processes used when creating
                the dataloader.

        Returns:
            cvtk.data.ImageDataset: Predicted images and annotations as an
            image dataset.
        """
        if isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            dataset = Dataset(self.datalabel, data, transform=DataTransform(is_train=False))
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        return self._predict_as_imagedataset(dataloader, cutoff=cutoff)


class SegmRunner(DetRunner):
    """segmentation runner built on top of DetRunner.
    
    Current only supports Mask R-CNN family of models.
    All methods and attributes are inherited from DetRunner.
    To use SegmRunner, simply replace DetRunner with SegmRunner in your code and refer to the documentation of DetRunner for usage.
    """

    def __init__(
        self,
        datalabel: cvtk_ml_data.DataLabel | str | list[str] | tuple[str],
        model: str | torch.nn.Module = 'maskrcnn_resnet50_fpn',
        weights: str | Any | None = None,
        workspace: str | None = None,
        device: str = 'auto'
    ) -> None:
        """Create a torchvision Mask R-CNN segmentation runner.

        Args:
            datalabel (cvtk_ml_data.DataLabel|str|list|tuple): A DataLabel,
                label file, or list of class names.
            model (str|torch.nn.Module): Torchvision segmentation model name or
                an instantiated module.
            weights (str|torchvision.models.detection weights|None): Optional
                weights enum name, checkpoint path, or weights object accepted
                by torchvision.
            workspace (str|None): Optional output directory for checkpoints and
                reports.
            device (str): Target device name or ``'auto'`` to pick CUDA when
                available.
        """
        super().__init__(datalabel=datalabel,
                         model=model,
                         weights=weights,
                         workspace=workspace,
                         device=device)
        self.task_type = 'segm'


    def _replace_detection_head(self, model: torch.nn.Module, arch_name: str, num_classes: int) -> torch.nn.Module:
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'box_predictor') and hasattr(model.roi_heads, 'mask_predictor'):
            box_in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)

            mask_in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
            mask_dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in_channels, mask_dim_reduced, num_classes)
            return model

        return super()._replace_detection_head(model, arch_name, num_classes)


    def _predict_as_imagedataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        cutoff: float = 0.5
    ) -> cvtk_data.ImageDataset:
        self.model.eval()
        records = []

        with torch.no_grad():
            for images, targets, paths in dataloader:
                images_device = [img.to(self.device, non_blocking=True) for img in images]
                outputs = self.model(images_device)

                for image_tensor, img_path, pred in zip(images, paths, outputs):
                    annotations = []
                    h = int(image_tensor.shape[-2])
                    w = int(image_tensor.shape[-1])
                    imsize = (w, h)

                    boxes = pred.get('boxes', torch.empty((0, 4)))
                    labels = pred.get('labels', torch.empty((0,), dtype=torch.int64))
                    scores = pred.get('scores', torch.empty((0,), dtype=torch.float32))
                    masks = pred.get('masks', torch.empty((0, 1, h, w), dtype=torch.float32))

                    boxes = boxes.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    scores = scores.detach().cpu().numpy()
                    masks = masks.detach().cpu().numpy()

                    for i, (box, label_id, score) in enumerate(zip(boxes, labels, scores)):
                        if float(score) < cutoff:
                            continue
                        label_index = int(label_id) - 1
                        if label_index < 0 or label_index >= len(self.datalabel.labels):
                            continue

                        bbox = cvtk_data.Bbox.from_xyxy(*box.tolist(), imsize=imsize)
                        segm = None
                        if i < len(masks):
                            mask_bin = (masks[i, 0] >= 0.5).astype(np.uint8)
                            if mask_bin.any():
                                segm = cvtk_data.Segm.from_mask(mask_bin)

                        ann = cvtk_data.InstanceAnnotation(
                            label=self.datalabel[label_index],
                            bbox=bbox,
                            segm=segm,
                            score=float(score),
                        )
                        annotations.append(ann)

                    records.append(
                        cvtk_data.ImageRecord(
                            source=pathlib.Path(img_path),
                            annotations=annotations,
                            size=imsize,
                        )
                    )

        return cvtk_data.ImageDataset(records=records)

