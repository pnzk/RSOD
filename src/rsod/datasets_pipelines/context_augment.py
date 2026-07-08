"""Context-augmented Copy-Paste for remote-sensing few-shot detection.

This transform implements the highest-leverage finding from recent DIOR
few-shot literature: remote-sensing detectors overfit *context* extremely
quickly in 3/5-shot regimes, so pasting scarce novel instances onto many
background scenes is often more useful than changing the detector head.

Unlike the earlier draft, this version returns a **single** training sample,
so it is compatible with the standard mmfewshot detection dataloader.
"""
from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mmdet.datasets import PIPELINES


BBox = Tuple[int, int, int, int]


def _clip_bbox(bbox: Sequence[float], w: int, h: int) -> BBox:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _mask_to_bbox(mask: np.ndarray) -> Optional[BBox]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    boxes = boxes.astype(np.float32, copy=False)
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h
    area_a = np.maximum(box[2] - box[0], 0.0) * np.maximum(box[3] - box[1], 0.0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return inter / union


def _random_affine_crop(
    img: np.ndarray,
    bbox: Sequence[float],
    rng: random.Random,
    context_px: int = 16,
    target_size: int = 224,
    max_rotate_deg: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, BBox]:
    """Crop one instance with context, apply light geometric jitter, and
    return a square crop + binary mask + bbox inside the crop.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    pad_x = max(context_px, int(round(0.05 * bw)))
    pad_y = max(context_px, int(round(0.05 * bh)))
    crop_size = max(bw + 2 * pad_x, bh + 2 * pad_y)
    crop_size = max(crop_size, 8)

    tx1 = int(round(cx - crop_size / 2.0))
    ty1 = int(round(cy - crop_size / 2.0))
    tx2 = tx1 + crop_size
    ty2 = ty1 + crop_size

    ox1 = max(tx1, 0)
    oy1 = max(ty1, 0)
    ox2 = min(tx2, w)
    oy2 = min(ty2, h)

    crop = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
    box_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    paste_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)

    crop[oy1 - ty1:oy2 - ty1, ox1 - tx1:ox2 - tx1] = img[oy1:oy2, ox1:ox2]
    # Set the area with actual image pixels inside the crop to 1 for pasting context
    paste_mask[oy1 - ty1:oy2 - ty1, ox1 - tx1:ox2 - tx1] = 1

    bx1 = max(x1 - tx1, 0)
    by1 = max(y1 - ty1, 0)
    bx2 = min(x2 - tx1, crop_size)
    by2 = min(y2 - ty1, crop_size)
    box_mask[by1:by2, bx1:bx2] = 1

    if rng.random() < 0.5:
        angle = rng.uniform(-max_rotate_deg, max_rotate_deg)
        center = (crop_size / 2.0, crop_size / 2.0)
        affine = cv2.getRotationMatrix2D(center, angle, 1.0)
        crop = cv2.warpAffine(
            crop,
            affine,
            (crop_size, crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101)
        box_mask = cv2.warpAffine(
            box_mask,
            affine,
            (crop_size, crop_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        paste_mask = cv2.warpAffine(
            paste_mask,
            affine,
            (crop_size, crop_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)

    if rng.random() < 0.5:
        crop = np.flip(crop, axis=1).copy()
        box_mask = np.flip(box_mask, axis=1).copy()
        paste_mask = np.flip(paste_mask, axis=1).copy()

    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    box_mask = cv2.resize(box_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    paste_mask = cv2.resize(paste_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    bbox_from_mask = _mask_to_bbox(box_mask)
    if bbox_from_mask is None:
        bbox_from_mask = (0, 0, target_size, target_size)
    return crop, paste_mask, bbox_from_mask


@PIPELINES.register_module()
class ContextCopyPasteAugment:
    """Paste novel-class instances onto fully-labeled base-only backgrounds.

    The transform is intentionally lightweight and detector-agnostic:
    no model code changes, only better fine-tuning samples.
    """

    def __init__(
        self,
        bg_dir: str = 'data/DIOR/JPEGImages',
        bg_ann_dir: str = 'data/DIOR/Annotations',
        all_classes: Optional[Sequence[str]] = None,
        novel_classes: Optional[Sequence[str]] = None,
        novel_label_ids: Optional[Sequence[int]] = None,
        coordinate_offset: Tuple[int, int, int, int] = (-1, -1, 0, 0),
        min_paste: int = 1,
        max_paste: int = 3,
        context_px: int = 16,
        target_size: int = 224,
        paste_scale: Tuple[float, float] = (0.8, 1.2),
        keep_original_prob: float = 0.25,
        base_only_bg: bool = True,
        allow_empty_bg: bool = True,
        max_bg_tries: int = 50,
        max_place_tries: int = 50,
        max_overlap_iou: float = 0.30,
        max_rotate_deg: float = 15.0,
        scene_class_compat: Optional[Dict[str, Sequence[str]]] = None,
        bg_format: str = 'voc_xml',
        # --- Leakage prevention parameters ---
        exclude_image_list: Optional[str] = None,
        allowed_image_list: Optional[str] = None,
        strip_novel_from_bg: bool = True,
        merge_bg_annotations: str = 'none',
        seed: int = 42,
    ) -> None:
        self.bg_dir = bg_dir
        self.bg_ann_dir = bg_ann_dir
        self.bg_format = str(bg_format)
        assert self.bg_format in ('voc_xml', 'nwpu_txt'), \
            f'Unknown bg_format: {self.bg_format}'
        self.all_classes = list(all_classes) if all_classes is not None else None
        self.novel_classes = list(novel_classes) if novel_classes is not None else []
        self.coordinate_offset = tuple(int(v) for v in coordinate_offset)
        self.min_paste = max(1, int(min_paste))
        self.max_paste = max(self.min_paste, int(max_paste))
        self.context_px = int(context_px)
        self.target_size = int(target_size)
        self.paste_scale = tuple(float(v) for v in paste_scale)
        self.keep_original_prob = float(keep_original_prob)
        self.base_only_bg = bool(base_only_bg)
        self.allow_empty_bg = bool(allow_empty_bg)
        self.max_bg_tries = int(max_bg_tries)
        self.max_place_tries = int(max_place_tries)
        self.max_overlap_iou = float(max_overlap_iou)
        self.max_rotate_deg = float(max_rotate_deg)
        self._rng = random.Random(seed)
        self._bg_pool: List[Dict] = []
        self._initialized = False

        # Leakage prevention:
        # exclude_image_list: path to txt file listing image IDs to exclude
        #   (e.g., test.txt) from background pool
        # allowed_image_list: path to txt file listing ONLY allowed image IDs
        #   (e.g., trainval.txt). If set, only these images can be backgrounds.
        # strip_novel_from_bg: if True, remove novel-class bboxes from
        #   background annotations before merging (prevents novel label leak)
        # merge_bg_annotations: 'none' = don't merge bg annotations into GT;
        #   'base_only' = merge only base-class annotations;
        #   'all' = merge all (LEGACY, causes leakage)
        self.exclude_image_list = exclude_image_list
        self.allowed_image_list = allowed_image_list
        self.strip_novel_from_bg = bool(strip_novel_from_bg)
        assert merge_bg_annotations in ('none', 'base_only', 'all'), \
            f'merge_bg_annotations must be none/base_only/all, got {merge_bg_annotations}'
        self.merge_bg_annotations = merge_bg_annotations

        # Load exclude/allow lists
        self._excluded_ids: Optional[set] = None
        self._allowed_ids: Optional[set] = None
        if self.exclude_image_list and os.path.isfile(self.exclude_image_list):
            with open(self.exclude_image_list, 'r') as f:
                self._excluded_ids = {line.strip() for line in f if line.strip()}
        if self.allowed_image_list and os.path.isfile(self.allowed_image_list):
            with open(self.allowed_image_list, 'r') as f:
                self._allowed_ids = {line.strip() for line in f if line.strip()}

        self.class_to_label = None
        if self.all_classes is not None:
            self.class_to_label = {name: idx for idx, name in enumerate(self.all_classes)}

        if novel_label_ids is not None:
            self.novel_label_ids = {int(v) for v in novel_label_ids}
        elif self.class_to_label is not None and self.novel_classes:
            self.novel_label_ids = {
                self.class_to_label[name]
                for name in self.novel_classes
                if name in self.class_to_label
            }
        else:
            self.novel_label_ids = set()
        self.novel_class_set = set(self.novel_classes)
        # Scene-class compatibility: maps novel class name -> set of base class
        # names that represent compatible background scenes. When set, the
        # background sampler prefers images containing at least one compatible
        # base class for the novel instance being pasted. This injects
        # remote-sensing domain knowledge (e.g. ship -> harbor/waterfront).
        if scene_class_compat is not None:
            self.scene_class_compat = {
                k: set(v) for k, v in scene_class_compat.items()
            }
        else:
            self.scene_class_compat = None

    def _parse_background_entry(self, ann_path: str) -> Optional[Dict]:
        img_id = Path(ann_path).stem

        # Check exclude/allow lists
        if self._excluded_ids is not None and img_id in self._excluded_ids:
            return None
        if self._allowed_ids is not None and img_id not in self._allowed_ids:
            return None

        img_path = os.path.join(self.bg_dir, f'{img_id}.jpg')
        if not os.path.isfile(img_path):
            return None

        try:
            root = ET.parse(ann_path).getroot()
        except Exception:
            return None

        labels: List[int] = []
        bboxes: List[List[float]] = []
        has_novel = False
        for obj in root.findall('object'):
            name = obj.find('name').text
            if self.novel_class_set and name in self.novel_class_set:
                has_novel = True
            if self.class_to_label is None or name not in self.class_to_label:
                continue
            label = self.class_to_label[name]
            # Skip novel-class bboxes if strip_novel_from_bg is True
            if self.strip_novel_from_bg and label in self.novel_label_ids:
                continue
            if self.base_only_bg and label in self.novel_label_ids:
                continue
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)) + self.coordinate_offset[0],
                int(float(bnd_box.find('ymin').text)) + self.coordinate_offset[1],
                int(float(bnd_box.find('xmax').text)) + self.coordinate_offset[2],
                int(float(bnd_box.find('ymax').text)) + self.coordinate_offset[3],
            ]
            labels.append(label)
            bboxes.append(bbox)

        if self.base_only_bg and has_novel:
            return None
        if (not self.allow_empty_bg) and len(bboxes) == 0:
            return None

        return dict(
            img_id=img_id,
            img_path=img_path,
            bboxes=np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
            labels=np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
        )

    def _parse_nwpu_txt_entry(self, ann_path: str) -> Optional[Dict]:
        """Parse NWPU VHR-10 txt-format annotation file.

        Format per line: (x1,y1),(x2,y2),class_id (1-indexed)
        """
        img_id = Path(ann_path).stem

        # Check exclude/allow lists
        if self._excluded_ids is not None and img_id in self._excluded_ids:
            return None
        if self._allowed_ids is not None and img_id not in self._allowed_ids:
            return None

        img_path = os.path.join(self.bg_dir, f'{img_id}.jpg')
        if not os.path.isfile(img_path):
            return None

        # NWPU class id (1-indexed) -> name mapping (mirrors VHR10_SPLIT.CLASSES)
        nwpu_id_to_name = (
            'airplane', 'ship', 'storage-tank', 'baseball-diamond',
            'tennis-court', 'basketball-court', 'ground-track-field',
            'harbor', 'bridge', 'vehicle')

        labels: List[int] = []
        bboxes: List[List[float]] = []
        has_novel = False
        try:
            with open(ann_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return None

        import re as _re
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = _re.match(
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*(\d+)',
                line)
            if not m:
                continue
            x1, y1, x2, y2, cls_id = map(int, m.groups())
            if cls_id < 1 or cls_id > len(nwpu_id_to_name):
                continue
            name = nwpu_id_to_name[cls_id - 1]
            if self.novel_class_set and name in self.novel_class_set:
                has_novel = True
            if self.class_to_label is None or name not in self.class_to_label:
                continue
            label = self.class_to_label[name]
            if self.base_only_bg and label in self.novel_label_ids:
                continue
            bbox = [
                x1 + self.coordinate_offset[0],
                y1 + self.coordinate_offset[1],
                x2 + self.coordinate_offset[2],
                y2 + self.coordinate_offset[3],
            ]
            labels.append(label)
            bboxes.append(bbox)

        if self.base_only_bg and has_novel:
            return None
        if (not self.allow_empty_bg) and len(bboxes) == 0:
            return None

        return dict(
            img_id=img_id,
            img_path=img_path,
            bboxes=np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
            labels=np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
        )

    def _ensure_bg_pool(self) -> None:
        if self._initialized:
            return
        ann_dir = Path(self.bg_ann_dir)
        if ann_dir.is_dir():
            if self.bg_format == 'nwpu_txt':
                # NWPU VHR-10 txt-format annotations (no extension or .txt)
                for ann_path in sorted(ann_dir.glob('*.txt')):
                    entry = self._parse_nwpu_txt_entry(str(ann_path))
                    if entry is not None:
                        self._bg_pool.append(entry)
            else:
                for ann_path in sorted(ann_dir.glob('*.xml')):
                    entry = self._parse_background_entry(str(ann_path))
                    if entry is not None:
                        self._bg_pool.append(entry)
        self._initialized = True

    def _sample_background(self, source_img_id: Optional[str] = None,
                           novel_class_name: Optional[str] = None) -> Optional[Dict]:
        self._ensure_bg_pool()
        if not self._bg_pool:
            return None
        # If scene-class compatibility is active and the novel class has a
        # preference, try to find a compatible background first.
        compat_classes = None
        if (self.scene_class_compat is not None and
                novel_class_name is not None and
                novel_class_name in self.scene_class_compat):
            compat_classes = self.scene_class_compat[novel_class_name]
        if compat_classes:
            compat_pool = []
            for entry in self._bg_pool:
                if source_img_id is not None and entry['img_id'] == source_img_id:
                    continue
                # Check if any label in this bg image belongs to a compatible class
                entry_classes = set()
                if self.all_classes is not None:
                    for lbl in entry['labels'].tolist():
                        if 0 <= lbl < len(self.all_classes):
                            entry_classes.add(self.all_classes[lbl])
                if entry_classes & compat_classes:
                    compat_pool.append(entry)
            if compat_pool:
                return self._rng.choice(compat_pool)
            # Fallback to random if no compatible bg found
        for _ in range(self.max_bg_tries):
            entry = self._rng.choice(self._bg_pool)
            if source_img_id is not None and entry['img_id'] == source_img_id:
                continue
            return entry
        return self._rng.choice(self._bg_pool)

    def _load_background(self, entry: Dict, target_h: int, target_w: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        bg = cv2.imread(entry['img_path'])
        if bg is None:
            return None
        bg_h, bg_w = bg.shape[:2]
        bboxes = entry['bboxes'].copy()
        labels = entry['labels'].copy()
        if bg_h != target_h or bg_w != target_w:
            scale_x = float(target_w) / max(float(bg_w), 1.0)
            scale_y = float(target_h) / max(float(bg_h), 1.0)
            if bboxes.size > 0:
                bboxes[:, 0] *= scale_x
                bboxes[:, 2] *= scale_x
                bboxes[:, 1] *= scale_y
                bboxes[:, 3] *= scale_y
            bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return bg, bboxes, labels

    def _choose_novel_indices(self, gt_labels: np.ndarray) -> List[int]:
        if not self.novel_label_ids:
            return list(range(len(gt_labels)))
        return [idx for idx, label in enumerate(gt_labels.tolist()) if int(label) in self.novel_label_ids]

    def _place_crop(
        self,
        bg: np.ndarray,
        crop: np.ndarray,
        mask: np.ndarray,
        crop_bbox: BBox,
        existing_boxes: np.ndarray,
    ) -> Optional[BBox]:
        h, w = bg.shape[:2]
        ch, cw = crop.shape[:2]
        scale = self._rng.uniform(*self.paste_scale)
        new_w = max(8, int(round(cw * scale)))
        new_h = max(8, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        bx1, by1, bx2, by2 = crop_bbox
        sx = float(new_w) / max(float(cw), 1.0)
        sy = float(new_h) / max(float(ch), 1.0)
        obj_w = max(int(round((bx2 - bx1) * sx)), 1)
        obj_h = max(int(round((by2 - by1) * sy)), 1)

        max_x = max(w - new_w, 0)
        max_y = max(h - new_h, 0)
        for _ in range(self.max_place_tries):
            px = self._rng.randint(0, max_x) if max_x > 0 else 0
            py = self._rng.randint(0, max_y) if max_y > 0 else 0
            new_box = np.array([
                px + int(round(bx1 * sx)),
                py + int(round(by1 * sy)),
                px + int(round(bx2 * sx)),
                py + int(round(by2 * sy)),
            ], dtype=np.float32)
            new_box[0] = max(0, min(new_box[0], w - 1))
            new_box[1] = max(0, min(new_box[1], h - 1))
            new_box[2] = max(0, min(new_box[2], w))
            new_box[3] = max(0, min(new_box[3], h))
            if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
                continue
            if existing_boxes.size > 0 and np.max(_bbox_iou(new_box, existing_boxes)) > self.max_overlap_iou:
                continue

            alpha = mask[..., None].astype(np.float32)
            patch = bg[py:py + new_h, px:px + new_w].astype(np.float32)
            crop_f = crop.astype(np.float32)
            bg[py:py + new_h, px:px + new_w] = (alpha * crop_f + (1.0 - alpha) * patch).astype(np.uint8)
            return tuple(int(v) for v in new_box.tolist())
        return None

    def __call__(self, results: Dict) -> Dict:
        img = results['img']
        gt_bboxes = results.get('gt_bboxes', np.zeros((0, 4), dtype=np.float32))
        gt_labels = results.get('gt_labels', np.zeros((0,), dtype=np.int64))
        if gt_bboxes is None or len(gt_bboxes) == 0:
            return results

        novel_indices = self._choose_novel_indices(gt_labels)
        if len(novel_indices) == 0:
            return results
        if self._rng.random() < self.keep_original_prob:
            return results

        source_img_id = None
        if results.get('img_info', None) is not None:
            source_img_id = results['img_info'].get('id', None)

        bg_entry = self._sample_background(
            source_img_id=source_img_id,
            novel_class_name=(
                self.all_classes[int(gt_labels[novel_indices[0]])]
                if (self.all_classes is not None and
                    self.scene_class_compat is not None and
                    len(novel_indices) > 0)
                else None))
        if bg_entry is None:
            return results

        target_h, target_w = img.shape[:2]
        loaded = self._load_background(bg_entry, target_h, target_w)
        if loaded is None:
            return results
        bg, bg_bboxes, bg_labels = loaded

        existing_boxes = bg_bboxes.copy()
        pasted_bboxes: List[List[float]] = []
        pasted_labels: List[int] = []

        num_select = min(len(novel_indices), self.max_paste)
        num_select = self._rng.randint(self.min_paste, max(self.min_paste, num_select))
        chosen_indices = self._rng.sample(novel_indices, min(num_select, len(novel_indices)))

        for idx in chosen_indices:
            crop, mask, crop_bbox = _random_affine_crop(
                img,
                gt_bboxes[idx],
                rng=self._rng,
                context_px=self.context_px,
                target_size=self.target_size,
                max_rotate_deg=self.max_rotate_deg)
            placed_bbox = self._place_crop(bg, crop, mask, crop_bbox, existing_boxes)
            if placed_bbox is None:
                continue
            pasted_bboxes.append(list(placed_bbox))
            pasted_labels.append(int(gt_labels[idx]))
            if existing_boxes.size == 0:
                existing_boxes = np.array([placed_bbox], dtype=np.float32)
            else:
                existing_boxes = np.concatenate(
                    [existing_boxes, np.array([placed_bbox], dtype=np.float32)], axis=0)

        if len(pasted_bboxes) == 0:
            return results

        # Merge annotations based on merge_bg_annotations policy
        if self.merge_bg_annotations == 'none':
            # Only use pasted novel instances as GT (no background annotations)
            merged_bboxes = np.array(pasted_bboxes, dtype=np.float32)
            merged_labels = np.array(pasted_labels, dtype=np.int64)
        elif self.merge_bg_annotations == 'base_only':
            # Merge only base-class annotations from background
            if bg_bboxes.size > 0:
                base_mask = np.array([
                    int(l) not in self.novel_label_ids
                    for l in bg_labels.tolist()
                ], dtype=bool)
                base_bboxes = bg_bboxes[base_mask] if base_mask.any() else np.zeros((0, 4), dtype=np.float32)
                base_labels = bg_labels[base_mask] if base_mask.any() else np.zeros((0,), dtype=np.int64)
                merged_bboxes = np.concatenate(
                    [base_bboxes, np.array(pasted_bboxes, dtype=np.float32)], axis=0)
                merged_labels = np.concatenate(
                    [base_labels, np.array(pasted_labels, dtype=np.int64)], axis=0)
            else:
                merged_bboxes = np.array(pasted_bboxes, dtype=np.float32)
                merged_labels = np.array(pasted_labels, dtype=np.int64)
        else:  # 'all' - legacy behavior (causes leakage!)
            if bg_bboxes.size > 0:
                merged_bboxes = np.concatenate(
                    [bg_bboxes, np.array(pasted_bboxes, dtype=np.float32)], axis=0)
                merged_labels = np.concatenate(
                    [bg_labels, np.array(pasted_labels, dtype=np.int64)], axis=0)
            else:
                merged_bboxes = np.array(pasted_bboxes, dtype=np.float32)
                merged_labels = np.array(pasted_labels, dtype=np.int64)

        results['img'] = bg
        results['img_shape'] = bg.shape
        results['ori_shape'] = bg.shape
        results['gt_bboxes'] = merged_bboxes.astype(np.float32)
        results['gt_labels'] = merged_labels.astype(np.int64)
        results['gt_bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
        results['gt_labels_ignore'] = np.zeros((0,), dtype=np.int64)
        if results.get('img_info', None) is not None:
            results['img_info'] = results['img_info'].copy()
            results['img_info']['filename'] = os.path.join('JPEGImages', f"{bg_entry['img_id']}.jpg")
            results['img_info']['id'] = f"{bg_entry['img_id']}__ctxaug"
        results['filename'] = bg_entry['img_path']
        results['ori_filename'] = f"{bg_entry['img_id']}__ctxaug.jpg"
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'min_paste={self.min_paste}, max_paste={self.max_paste}, '
            f'context_px={self.context_px}, target_size={self.target_size}, '
            f'keep_original_prob={self.keep_original_prob}, '
            f'base_only_bg={self.base_only_bg})'
        )


@PIPELINES.register_module()
class RemoteSensingCutMixAugment(ContextCopyPasteAugment):
    """CutMix-style rectangular mixing for detection ablations.

    This transform copies a random rectangular region from a background image
    into the current training image, keeps source boxes outside the mixed
    rectangle, and adds clipped background boxes whose visible area is large
    enough. It is intentionally label-agnostic and context-unaware, making it a
    controlled counterpart to ContextCopyPasteAugment in remote-sensing FSOD
    ablation experiments.
    """

    def __init__(
        self,
        prob: float = 1.0,
        area_ratio: Tuple[float, float] = (0.15, 0.50),
        aspect_ratio: Tuple[float, float] = (0.50, 2.00),
        min_box_visibility: float = 0.30,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prob = float(prob)
        self.area_ratio = tuple(float(v) for v in area_ratio)
        self.aspect_ratio = tuple(float(v) for v in aspect_ratio)
        self.min_box_visibility = float(min_box_visibility)

    def _sample_rect(self, h: int, w: int) -> BBox:
        area = max(float(h * w), 1.0)
        ratio = self._rng.uniform(*self.area_ratio)
        aspect = self._rng.uniform(*self.aspect_ratio)
        cut_w = int(round((area * ratio * aspect) ** 0.5))
        cut_h = int(round((area * ratio / max(aspect, 1e-6)) ** 0.5))
        cut_w = max(8, min(cut_w, w))
        cut_h = max(8, min(cut_h, h))
        x1 = self._rng.randint(0, max(w - cut_w, 0)) if w > cut_w else 0
        y1 = self._rng.randint(0, max(h - cut_h, 0)) if h > cut_h else 0
        return x1, y1, x1 + cut_w, y1 + cut_h

    @staticmethod
    def _clip_boxes_to_rect(boxes: np.ndarray, rect: BBox) -> Tuple[np.ndarray, np.ndarray]:
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32), np.zeros((0,), dtype=bool)
        x1, y1, x2, y2 = rect
        clipped = boxes.astype(np.float32, copy=True)
        clipped[:, 0] = np.maximum(clipped[:, 0], x1)
        clipped[:, 1] = np.maximum(clipped[:, 1], y1)
        clipped[:, 2] = np.minimum(clipped[:, 2], x2)
        clipped[:, 3] = np.minimum(clipped[:, 3], y2)
        valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
        return clipped, valid

    def __call__(self, results: Dict) -> Dict:
        img = results['img']
        h, w = img.shape[:2]
        if self._rng.random() > self.prob:
            return results

        source_img_id = None
        if results.get('img_info', None) is not None:
            source_img_id = results['img_info'].get('id', None)
        bg_entry = self._sample_background(source_img_id=source_img_id)
        if bg_entry is None:
            return results
        loaded = self._load_background(bg_entry, h, w)
        if loaded is None:
            return results
        bg, bg_bboxes, bg_labels = loaded

        rect = self._sample_rect(h, w)
        x1, y1, x2, y2 = rect
        mixed = img.copy()
        mixed[y1:y2, x1:x2] = bg[y1:y2, x1:x2]

        src_bboxes = results.get('gt_bboxes', np.zeros((0, 4), dtype=np.float32))
        src_labels = results.get('gt_labels', np.zeros((0,), dtype=np.int64))
        if src_bboxes is None:
            src_bboxes = np.zeros((0, 4), dtype=np.float32)
        if src_labels is None:
            src_labels = np.zeros((0,), dtype=np.int64)

        # Remove source boxes whose centers fall inside the replaced rectangle.
        keep_src = np.ones((len(src_bboxes),), dtype=bool)
        if len(src_bboxes) > 0:
            centers_x = 0.5 * (src_bboxes[:, 0] + src_bboxes[:, 2])
            centers_y = 0.5 * (src_bboxes[:, 1] + src_bboxes[:, 3])
            inside = (centers_x >= x1) & (centers_x <= x2) & (centers_y >= y1) & (centers_y <= y2)
            keep_src = ~inside

        clipped_bg, valid_bg = self._clip_boxes_to_rect(bg_bboxes, rect)
        if bg_bboxes.size > 0:
            orig_area = np.maximum(bg_bboxes[:, 2] - bg_bboxes[:, 0], 0.0) * \
                np.maximum(bg_bboxes[:, 3] - bg_bboxes[:, 1], 0.0)
            clip_area = np.maximum(clipped_bg[:, 2] - clipped_bg[:, 0], 0.0) * \
                np.maximum(clipped_bg[:, 3] - clipped_bg[:, 1], 0.0)
            visible = clip_area / np.maximum(orig_area, 1e-6)
            valid_bg = valid_bg & (visible >= self.min_box_visibility)

        merged_bboxes = src_bboxes[keep_src].astype(np.float32, copy=False)
        merged_labels = src_labels[keep_src].astype(np.int64, copy=False)
        if bg_bboxes.size > 0 and valid_bg.any():
            merged_bboxes = np.concatenate([merged_bboxes, clipped_bg[valid_bg]], axis=0)
            merged_labels = np.concatenate([merged_labels, bg_labels[valid_bg]], axis=0)

        results['img'] = mixed
        results['img_shape'] = mixed.shape
        results['gt_bboxes'] = merged_bboxes.astype(np.float32)
        results['gt_labels'] = merged_labels.astype(np.int64)
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'prob={self.prob}, area_ratio={self.area_ratio}, '
            f'aspect_ratio={self.aspect_ratio}, base_only_bg={self.base_only_bg})'
        )
