"""
detectors/dataset_builder.py
-----------------------------
Build a YOLO-format dataset from raw OCT images + generated bounding boxes.

Responsibilities
----------------
1. Reproducibly split filenames into train / val / test sets.
2. Copy images to ``data/roi_dataset/images/{train,val,test}/``.
3. Write YOLO label .txt files to ``data/roi_dataset/labels/{train,val,test}/``.
4. Generate ``dataset.yaml`` required by ultralytics YOLOv8.
5. Produce a split manifest CSV for traceability.

Design decisions
----------------
- Images are **copied** (not symlinked) so the dataset is self-contained
  and Colab can read it from local SSD without Drive I/O overhead.
- YOLOv8 expects 3-channel images; greyscale OCT images are converted to
  pseudo-RGB (cv2.MERGE of 3 identical channels) during the copy step.
  This avoids modifying any training-time code.
- Images with no lesion boxes are excluded from the YOLO dataset by default
  (``include_negatives=False``). They contain no useful detection signal.
  Set to True to include them as hard negatives.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from detectors.bbox_generator import BoundingBox, write_yolo_label

logger = logging.getLogger(__name__)


# ── Train / Val / Test split ───────────────────────────────────────────────────

def split_filenames(
    filenames: List[str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly split a filename list into train / val / test sets.

    Parameters
    ----------
    filenames   : list of image filenames.
    train_ratio : fraction for training.
    val_ratio   : fraction for validation (test = 1 - train - val).
    seed        : random seed for reproducibility.

    Returns
    -------
    train_files, val_files, test_files
    """
    rng = random.Random(seed)
    shuffled = filenames.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio   * n)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    logger.info(
        "Split → Train: %d | Val: %d | Test: %d (total: %d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test


# ── Dataset writer ─────────────────────────────────────────────────────────────

def build_yolo_dataset(
    image_dir:          str,
    bbox_results:       Dict[str, List[BoundingBox]],
    output_root:        str,
    train_ratio:        float = 0.70,
    val_ratio:          float = 0.15,
    seed:               int   = 42,
    include_negatives:  bool  = False,
    class_names:        Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Write a complete YOLO-format dataset to disk.

    Parameters
    ----------
    image_dir          : source image directory.
    bbox_results       : ``{filename: List[BoundingBox]}`` from bbox_generator.
    output_root        : root dir for YOLO dataset (e.g. "data/roi_dataset").
    train_ratio        : training split fraction.
    val_ratio          : validation split fraction.
    seed               : random seed.
    include_negatives  : if True, include images with no boxes.
    class_names        : YOLO class names (default: ["lesion"]).

    Returns
    -------
    dict ``{"train": [...], "val": [...], "test": [...]}``  — filename lists.
    """
    class_names = class_names or ["lesion"]
    output_root = Path(output_root)

    # ── Create directory structure ────────────────────────────────────────────
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Filter: only files with at least one bbox (unless negatives wanted) ──
    eligible = [
        f for f, boxes in bbox_results.items()
        if boxes or include_negatives
    ]
    n_excluded = len(bbox_results) - len(eligible)
    if n_excluded:
        logger.info("Excluded %d images with no lesion boxes.", n_excluded)

    # ── Split ─────────────────────────────────────────────────────────────────
    train_files, val_files, test_files = split_filenames(
        eligible, train_ratio, val_ratio, seed
    )
    splits = {"train": train_files, "val": val_files, "test": test_files}

    # ── Copy images + write labels ────────────────────────────────────────────
    for split_name, files in splits.items():
        img_out_dir = output_root / "images" / split_name
        lbl_out_dir = output_root / "labels" / split_name

        for fname in files:
            src_img = os.path.join(image_dir, fname)

            # Convert greyscale OCT → 3-channel PNG for YOLO
            img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Cannot read %s — skipping.", src_img)
                continue
            img_rgb = cv2.merge([img, img, img])   # pseudo-RGB

            stem    = Path(fname).stem
            out_img = img_out_dir / f"{stem}.png"
            cv2.imwrite(str(out_img), img_rgb)

            # Write label
            h, w    = img.shape
            boxes   = bbox_results.get(fname, [])
            lbl_path = str(lbl_out_dir / f"{stem}.txt")
            write_yolo_label(boxes, lbl_path, img_w=w, img_h=h)

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    abs_root = str(output_root.resolve())
    dataset_yaml = {
        "path":  abs_root,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info("dataset.yaml written → %s", yaml_path)

    # ── Write split manifest CSV ──────────────────────────────────────────────
    manifest_path = output_root / "split_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "num_boxes"])
        for split_name, files in splits.items():
            for fname in files:
                n_boxes = len(bbox_results.get(fname, []))
                writer.writerow([fname, split_name, n_boxes])
    logger.info("Split manifest → %s", manifest_path)

    return {k: v for k, v in splits.items()}


# ── Summary printer ────────────────────────────────────────────────────────────

def print_dataset_summary(
    splits: Dict[str, List[str]],
    bbox_results: Dict[str, List[BoundingBox]],
) -> None:
    """Print a concise dataset summary table."""
    sep = "─" * 50
    print(f"\n{sep}")
    print("  YOLO DATASET SUMMARY")
    print(sep)
    for split, files in splits.items():
        n_with = sum(1 for f in files if bbox_results.get(f))
        n_box  = sum(len(bbox_results.get(f, [])) for f in files)
        print(f"  {split:<8}: {len(files):>4} images | "
              f"{n_with:>4} with boxes | {n_box:>4} total boxes")
    total = sum(len(v) for v in splits.values())
    print(f"  {'TOTAL':<8}: {total:>4} images")
    print(sep + "\n")
