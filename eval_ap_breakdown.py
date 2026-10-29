#!/usr/bin/env python3
"""Per-class and per-object-size AP breakdown for Ultralytics YOLO detectors.

Purpose
-------
Reviewer comment 5 (IMAVIS-D-26-02785) asks for (i) per-category AP and
(ii) AP split by object size (e.g. objects smaller than 16x16 px). Ultralytics'
`model.val()` reports per-class AP but no size breakdown, so this script:

  1. runs the official Ultralytics validator and captures its exact (unscaled)
     predictions and ground truth,
  2. computes AP with the same `ap_per_class` protocol as `model.val()`, so the
     `all` row reproduces Table II exactly,
  3. derives per-size bins by matching once per IoU threshold and using COCO-style
     ignore semantics for GT outside the bin,
  4. writes per-class, per-size, and per-class x per-size tables (CSV + Markdown).

Default size bins (area in px^2):
    tiny    [0,      16^2)      < 16x16 px
    small   [16^2,   32^2)      16x16 - 32x32 px
    medium  [32^2,   96^2)      32x32 - 96x96 px
    large   [96^2,   inf)       > 96x96 px
    COCO-small [0, 32^2)        the standard COCO "small" range (for reference)

Size can be measured in two spaces (--area-space, default 'both'):
    original : annotation pixels as stored in the dataset (COCO convention)
    input    : pixels after letterbox resize to --imgsz (matches the paper's
               "12x12 object -> 4-9 tokens at stride 4" discussion)
For VisDrone, whose source images are much larger than 640, these two
conventions give very different tiny/small counts, so both are reported.

Usage
-----
  # single model
  python eval_ap_breakdown.py --weights runs/train/exp/weights/best.pt

  # compare several models in one go
  python eval_ap_breakdown.py \
      --weights out/ms-yolo-m/weights/best.pt out/yolo12-m/weights/best.pt \
      --tags ms-yolo-m yolo12-m \
      --data ultralytics/cfg/datasets/VisDrone.yaml \
      --device 0 --out runs/ap_breakdown

Notes
-----
* Uses the same evaluation settings as Ultralytics validation by default:
  conf=0.001, NMS iou=0.7, max_det=300, 101-point interpolated AP.
* Raise `--max-det` (e.g. 1000) if you want to reduce truncation in very
  crowded VisDrone images, but note this no longer matches `model.val()`.
* `--split` can be val / test / train (any key in the dataset YAML).
* Mamba-based models (MS-YOLO) use a custom CUDA selective-scan kernel and
  therefore need a GPU: pass `--device 0`. YOLOv12/RT-DETR run on CPU too.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pycocotools is required. Install it with:\n"
        "  pip install pycocotools"
    ) from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# area ranges in px^2
AREA_RANGES = [
    ("all", 0.0, 1e10),
    ("tiny(<16^2)", 0.0, 16**2),
    ("small(16^2-32^2)", 16**2, 32**2),
    ("medium(32^2-96^2)", 32**2, 96**2),
    ("large(>96^2)", 96**2, 1e10),
    ("COCO-small(<32^2)", 0.0, 32**2),
]


# --------------------------------------------------------------------------- #
# dataset helpers
# --------------------------------------------------------------------------- #
def resolve_dataset(data_yaml: str | Path, split: str):
    """Resolve dataset root + split path + class names from a YOLO data YAML."""
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")

    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = d.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}
    if not names:
        raise ValueError(f"'names' missing in {data_yaml}")

    root = Path(os.path.expandvars(str(d.get("path", "."))))
    if not root.is_absolute():
        candidates = [Path.cwd() / root, data_yaml.parent / root]
        candidates += [p / root for p in data_yaml.parents]
        root = next((c for c in candidates if c.exists()), candidates[0])
    root = root.resolve()

    if split not in d:
        raise KeyError(f"split '{split}' not found in {data_yaml} (keys: {list(d)})")
    split_path = Path(os.path.expandvars(str(d[split])))
    if not split_path.is_absolute():
        split_path = root / split_path

    if split_path.is_dir():
        images = sorted(p for p in split_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    elif split_path.is_file():
        if split_path.suffix.lower() in {".txt", ".csv"}:
            lines = [x.strip() for x in split_path.read_text().splitlines() if x.strip()]
            images = [Path(x) if Path(x).is_absolute() else root / x for x in lines]
        else:
            images = [split_path]
    else:
        raise FileNotFoundError(f"split path does not exist: {split_path}")

    if not images:
        raise FileNotFoundError(f"no images found for split '{split}' at {split_path}")
    return root, images, names


def image_to_label_path(img_path: Path) -> Path:
    """Convert .../images/.../x.jpg -> .../labels/.../x.txt (Ultralytics layout)."""
    parts = list(img_path.parts)
    try:
        idx = max(i for i, p in enumerate(parts) if p == "images")
        parts[idx] = "labels"
    except ValueError:
        # fallback: sibling 'labels' directory
        parts[-2] = "labels"
    return Path(*parts).with_suffix(".txt")


# --------------------------------------------------------------------------- #
# COCO ground truth
# --------------------------------------------------------------------------- #
def build_gt(images: list[Path], names: dict[int, str], imgsz: int = 640):
    """Build COCO-style GT dict from YOLO label files.

    Each annotation stores `area_original` (dataset pixels) and `area_input`
    (pixels after letterbox resize so that max(H, W) == imgsz).
    """
    coco = {
        "info": {"description": "built by eval_ap_breakdown.py"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": cid + 1, "name": name, "supercategory": "object"} for cid, name in sorted(names.items())],
    }
    path_to_id: dict[str, int] = {}
    ann_id = 0
    missing = 0

    for image_id, img_path in enumerate(images, start=1):
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cannot read {img_path}: {exc}", file=sys.stderr)
            continue

        coco["images"].append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )
        path_to_id[str(img_path.resolve())] = image_id

        label_path = image_to_label_path(img_path)
        if not label_path.is_file():
            missing += 1
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])

            # normalized YOLO -> absolute xywh, clipped to image bounds
            x1 = max(0.0, (cx - bw / 2.0) * width)
            y1 = max(0.0, (cy - bh / 2.0) * height)
            x2 = min(float(width), (cx + bw / 2.0) * width)
            y2 = min(float(height), (cy + bh / 2.0) * height)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            scale = float(imgsz) / max(width, height)
            ann_id += 1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,  # YOLO 0-based -> COCO 1-based
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "area_original": w * h,
                    "area_input": (w * scale) * (h * scale),
                    "iscrowd": 0,
                }
            )

    if missing:
        print(f"[warn] {missing} images have no label file (treated as background).", file=sys.stderr)
    return coco, path_to_id


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def run_inference(weights: str, images: list[Path], path_to_id: dict[str, int], args, names: dict[int, str]):
    """Run Ultralytics prediction and return COCO-format detections."""
    from ultralytics import YOLO

    model = YOLO(weights)
    model_names = getattr(model.model, "names", None)
    if model_names is not None and len(model_names) != len(names):
        print(
            f"[warn] model has {len(model_names)} classes but dataset has {len(names)}; "
            "predicted category ids may not align with the GT.",
            file=sys.stderr,
        )

    use_half = (not args.fp32) and str(args.device).lower() != "cpu"
    preds: list[dict] = []
    t0 = time.time()
    n_boxes = 0
    # NOTE: run one image at a time. In this Ultralytics fork, passing a list of
    # paths with stream=True batches ALL images into one tensor (e.g. 548x3x640x640),
    # which OOMs inside the first SOPE convolution on a 16 GB GPU.
    for k, img_path in enumerate(images, start=1):
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            half=use_half,
            verbose=False,
            save=False,
            show=False,
        )
        for r in results:
            img_id = path_to_id.get(str(Path(r.path).resolve()))
            if img_id is None:
                continue
            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.xyxy.numel() == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
                w, h = float(x2 - x1), float(y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                preds.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(cls) + 1,
                        "bbox": [float(x1), float(y1), w, h],
                        "score": float(score),
                    }
                )
                n_boxes += 1
        if k % 100 == 0:
            print(f"  ... {k}/{len(images)} images", flush=True)
    print(f"[info] inference: {len(images)} images, {n_boxes} boxes, {time.time() - t0:.1f}s")
    return preds


# --------------------------------------------------------------------------- #
# Ultralytics-protocol evaluation (matches `model.val()` exactly)
# --------------------------------------------------------------------------- #
IOUV = np.arange(0.5, 1.0, 0.05)  # 10 IoU thresholds


def run_validation(weights: str, args, out_dir: Path, tag: str):
    """Run the official validator and capture its *unscaled* predictions/GT.

    Table II's AP is computed in the letterboxed model-input space (both
    predictions and GT are xyxy in ``imgsz`` coordinates).  Exporting
    ``predictions.json`` scales boxes back to original image coordinates, and
    that small offset disproportionately hurts small objects.  We therefore
    hook ``DetectionValidator.update_metrics`` and keep the exact tensors used
    by the official metric.
    """
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator

    captured: list[dict] = []
    orig_update = DetectionValidator.update_metrics

    def _patched(self, preds, batch):
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            pn = self._prepare_pred(pred)
            captured.append(
                {
                    "pred_boxes": pn["bboxes"].detach().cpu().numpy(),
                    "pred_conf": pn["conf"].detach().cpu().numpy(),
                    "pred_cls": pn["cls"].detach().cpu().numpy().astype(int),
                    "gt_boxes": pb["bboxes"].detach().cpu().numpy(),
                    "gt_cls": pb["cls"].detach().cpu().numpy().astype(int),
                    "imgsz": tuple(int(x) for x in pb["imgsz"]),
                    "ori_shape": tuple(int(x) for x in pb["ori_shape"]),
                    "ratio_pad": pb["ratio_pad"],
                    "im_file": pb["im_file"],
                }
            )
        return orig_update(self, preds, batch)

    DetectionValidator.update_metrics = _patched
    try:
        model = YOLO(weights)
        r = model.val(
            data=str(Path(args.data).resolve()),
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            half=(not args.fp32) and str(args.device).lower() != "cpu",
            plots=False,
            verbose=False,
            project=str(out_dir / "_val_json"),
            name=tag.replace(os.sep, "_"),
            exist_ok=True,
        )
    finally:
        DetectionValidator.update_metrics = orig_update
    return captured, float(r.box.map50) * 100, float(r.box.map) * 100


def _greedy_match(iou: np.ndarray, thr: float) -> np.ndarray:
    """Greedy one-to-one matching, same rule as Ultralytics ``match_predictions``."""
    G, P = iou.shape
    matched = np.full(P, -1, dtype=int)
    if G == 0 or P == 0:
        return matched
    matches = np.argwhere(iou >= thr)
    if matches.size == 0:
        return matched
    vals = iou[matches[:, 0], matches[:, 1]]
    matches = matches[np.argsort(-vals)]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    matched[matches[:, 1]] = matches[:, 0]
    return matched


def evaluate_ultralytics(captured: list[dict], names: dict[int, str]):
    """Per-class and per-size AP using the official validator's matching.

    For the ``all`` bin this reproduces ``model.val()`` exactly (same tensors,
    same ``ap_per_class`` call).  For a size bin, a detection is a TP if its
    matched GT lies in the bin, is *ignored* if it matched a GT outside the bin
    (COCO-style), and an FP if unmatched.
    """
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import ap_per_class, box_iou

    acc = {}
    for space in ("original", "input"):
        for label, lo, hi in AREA_RANGES:
            acc[(space, label)] = {"tp": [], "ign": [], "conf": [], "pred_cls": [], "target_cls": []}

    for c in captured:
        gb = torch.as_tensor(c["gt_boxes"], dtype=torch.float32).reshape(-1, 4)
        pb = torch.as_tensor(c["pred_boxes"], dtype=torch.float32).reshape(-1, 4)
        gc = np.asarray(c["gt_cls"], dtype=int).reshape(-1)
        pc = np.asarray(c["pred_cls"], dtype=int).reshape(-1)
        conf = np.asarray(c["pred_conf"], dtype=float).reshape(-1)

        if len(gb) and len(pb):
            iou = box_iou(gb, pb).cpu().numpy()
            iou = iou * (gc[:, None] == pc[None, :])
        else:
            iou = np.zeros((len(gb), len(pb)))

        matched = (
            np.stack([_greedy_match(iou, t) for t in IOUV], axis=1)
            if len(pb)
            else np.zeros((0, len(IOUV)), dtype=int)
        )
        m_ok = matched >= 0

        areas = {
            "input": (gb[:, 2] - gb[:, 0]).numpy() * (gb[:, 3] - gb[:, 1]).numpy()
            if len(gb)
            else np.zeros(0)
        }
        if len(gb):
            gt_orig = ops.scale_boxes(c["imgsz"], gb.clone(), c["ori_shape"], ratio_pad=c["ratio_pad"]).numpy()
            areas["original"] = (gt_orig[:, 2] - gt_orig[:, 0]) * (gt_orig[:, 3] - gt_orig[:, 1])
        else:
            areas["original"] = np.zeros(0)

        for space in ("original", "input"):
            for label, lo, hi in AREA_RANGES:
                gt_in = np.ones(len(gb), bool) if label == "all" else (areas[space] >= lo) & (areas[space] < hi)
                in_bin = np.zeros_like(m_ok)
                if len(gb):
                    in_bin[m_ok] = gt_in[matched[m_ok]]
                a = acc[(space, label)]
                a["tp"].append(in_bin)
                a["ign"].append(m_ok & ~in_bin)
                a["conf"].append(conf)
                a["pred_cls"].append(pc)
                a["target_cls"].append(gc[gt_in])

    bin_rows = {}
    for key, a in acc.items():
        tp = np.concatenate(a["tp"], 0) if a["tp"] else np.zeros((0, len(IOUV)), bool)
        ign = np.concatenate(a["ign"], 0) if a["ign"] else np.zeros((0, len(IOUV)), bool)
        conf = np.concatenate(a["conf"], 0) if a["conf"] else np.zeros((0,))
        pcls = np.concatenate(a["pred_cls"], 0) if a["pred_cls"] else np.zeros((0,), int)
        tcls = np.concatenate(a["target_cls"], 0) if a["target_cls"] else np.zeros((0,), int)

        cls_ap = {int(c): np.full(len(IOUV), np.nan) for c in np.unique(tcls)}
        if len(tcls) and len(tp):
            if key[1] == "all":
                # exact official call: one (N, 10) matrix
                out = ap_per_class(tp, conf, pcls, tcls, plot=False, names=names)
                ap, ucls = out[5], out[6]
                for ci, c in enumerate(ucls):
                    cls_ap[int(c)] = ap[ci].astype(float)
            else:
                for ti in range(len(IOUV)):
                    keep = ~ign[:, ti]
                    tp_t = tp[keep, ti:ti + 1]
                    conf_t = conf[keep]
                    pcls_t = pcls[keep]
                    if len(tp_t) == 0:
                        for c in cls_ap:
                            cls_ap[c][ti] = 0.0
                        continue
                    out = ap_per_class(tp_t, conf_t, pcls_t, tcls, plot=False, names=names)
                    ap, ucls = out[5], out[6]
                    for ci, c in enumerate(ucls):
                        cls_ap[int(c)][ti] = float(ap[ci, 0])

        rows = []
        for c in sorted(cls_ap):
            vals = cls_ap[c]
            rows.append(
                {
                    "category_id": c + 1,
                    "class": names.get(c, str(c)),
                    "area_range": key[1],
                    "n_gt": int((tcls == c).sum()),
                    "AP50": float(vals[0] * 100) if np.isfinite(vals[0]) else float("nan"),
                    "AP50_95": float(np.nanmean(vals) * 100) if np.isfinite(vals).any() else float("nan"),
                }
            )
        bin_rows[key] = rows

    per_class = [
        {k: v for k, v in r.items() if k != "area_range"}
        for r in bin_rows[("original", "all")]
    ]

    size_tables, class_size_tables = {}, {}
    for space in ("original", "input"):
        ps, pcs = [], []
        for label, _, _ in AREA_RANGES:
            rows = bin_rows[(space, label)]
            pcs.extend(rows)
            good = [r for r in rows if np.isfinite(r["AP50_95"])]
            ps.append(
                {
                    "area_range": label,
                    "n_gt": sum(r["n_gt"] for r in rows),
                    "AP50": float(np.mean([r["AP50"] for r in good])) if good else float("nan"),
                    "AP50_95": float(np.mean([r["AP50_95"] for r in good])) if good else float("nan"),
                    "n_classes": len(good),
                }
            )
        size_tables[space] = ps
        class_size_tables[space] = pcs

    return per_class, size_tables, class_size_tables


# --------------------------------------------------------------------------- #
# COCO evaluation (legacy reference; not used by main)
# --------------------------------------------------------------------------- #
def evaluate_coco(gt: dict, preds: list[dict], names: dict[int, str], max_det: int,
                  area_key: str = "area_original"):
    """Run COCOeval with custom area ranges and return tidy tables."""
    # COCOeval filters by ann['area']; swap in the requested size space.
    gt_view = {
        **gt,
        "annotations": [{**a, "area": float(a[area_key])} for a in gt["annotations"]],
    }
    coco_gt = COCO()
    coco_gt.dataset = gt_view
    coco_gt.createIndex()

    if not preds:
        raise RuntimeError("no detections produced; check the model / conf threshold")

    coco_dt = coco_gt.loadRes(preds)

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.areaRng = [[lo, hi] for _, lo, hi in AREA_RANGES]
    ev.params.areaRngLbl = [label for label, _, _ in AREA_RANGES]
    ev.params.maxDets = [1, 10, max_det]
    ev.params.imgIds = sorted(coco_gt.getImgIds())
    ev.evaluate()
    ev.accumulate()

    precision = ev.eval["precision"]  # [T, R, K, A, M]
    cat_ids = ev.params.catIds
    cat_names = {cid: names.get(cid - 1, str(cid)) for cid in cat_ids}

    # GT counts per class / area
    img_ids = set(ev.params.imgIds)
    anns = [a for a in gt_view["annotations"] if a["image_id"] in img_ids]
    gt_counts = defaultdict(int)
    for a in anns:
        cid, area = a["category_id"], a["area"]
        gt_counts[(cid, "all")] += 1
        for label, lo, hi in AREA_RANGES[1:]:
            if lo <= area < hi:
                gt_counts[(cid, label)] += 1

    def ap_from_prec(p: np.ndarray):
        valid = p[p > -1]
        return float(valid.mean()) if valid.size else float("nan")

    per_class, per_size, per_class_size = [], [], []
    for ci, cid in enumerate(cat_ids):
        ap_all, ap50_all = None, None
        for ai, label in enumerate(ev.params.areaRngLbl):
            p = precision[:, :, ci, ai, -1]
            ap5095 = ap_from_prec(p)
            ap50 = ap_from_prec(p[0, :])
            n_gt = gt_counts.get((cid, label), 0)
            row = {
                "category_id": int(cid),
                "class": cat_names[cid],
                "area_range": label,
                "n_gt": n_gt,
                "AP50": ap50 * 100 if np.isfinite(ap50) else float("nan"),
                "AP50_95": ap5095 * 100 if np.isfinite(ap5095) else float("nan"),
            }
            per_class_size.append(row)
            if label == "all":
                ap_all, ap50_all = row["AP50_95"], row["AP50"]
        per_class.append(
            {
                "category_id": int(cid),
                "class": cat_names[cid],
                "n_gt": gt_counts.get((cid, "all"), 0),
                "AP50": ap50_all,
                "AP50_95": ap_all,
            }
        )

    # per-size aggregates: average over classes with valid AP, weighted GT counts for info
    for label in ev.params.areaRngLbl:
        rows = [r for r in per_class_size if r["area_range"] == label and np.isfinite(r["AP50_95"])]
        n_gt = sum(gt_counts.get((cid, label), 0) for cid in cat_ids)
        per_size.append(
            {
                "area_range": label,
                "n_gt": n_gt,
                "AP50": float(np.mean([r["AP50"] for r in rows])) if rows else float("nan"),
                "AP50_95": float(np.mean([r["AP50_95"] for r in rows])) if rows else float("nan"),
                "n_classes": len(rows),
            }
        )

    return per_class, per_size, per_class_size, ev


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


SIZE_SPACE_LABEL = {
    "original": "original annotation pixels",
    "input": "pixels after resize to model input",
}


def write_markdown(path: Path, tag: str, per_class: list[dict],
                   per_size_tables: dict[str, list[dict]], meta: dict):
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"

    lines = [
        f"# AP breakdown: {tag}",
        "",
        f"- weights: `{meta['weights']}`",
        f"- data: `{meta['data']}` / split `{meta['split']}`",
        f"- imgsz={meta['imgsz']}, conf={meta['conf']}, nms_iou={meta['iou']}, max_det={meta['max_det']}",
        f"- images={meta['n_images']}, GT boxes={meta['n_gt']}, detections={meta['n_det']}",
        "",
    ]
    for space, per_size in per_size_tables.items():
        lines += [
            f"## Per object size ({SIZE_SPACE_LABEL.get(space, space)})",
            "",
            "| area range | #GT | #classes | AP50 | AP50-95 |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in per_size:
            lines.append(
                f"| {r['area_range']} | {r['n_gt']} | {r['n_classes']} | {fmt(r['AP50'])} | {fmt(r['AP50_95'])} |"
            )
        lines.append("")

    lines += [
        "## Per class (all sizes)",
        "",
        "| id | class | #GT | AP50 | AP50-95 |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in per_class:
        lines.append(
            f"| {r['category_id']} | {r['class']} | {r['n_gt']} | {fmt(r['AP50'])} | {fmt(r['AP50_95'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(tag: str, per_class: list[dict], per_size_tables: dict[str, list[dict]]):
    print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")
    for space, per_size in per_size_tables.items():
        print(f"Per-size AP ({SIZE_SPACE_LABEL.get(space, space)}):")
        print(f"  {'area range':<22s} {'#GT':>8s} {'AP50':>8s} {'AP50-95':>9s}")
        for r in per_size:
            a50 = f"{r['AP50']:.2f}" if np.isfinite(r["AP50"]) else "n/a"
            a95 = f"{r['AP50_95']:.2f}" if np.isfinite(r["AP50_95"]) else "n/a"
            print(f"  {r['area_range']:<22s} {r['n_gt']:>8d} {a50:>8s} {a95:>9s}")
        print()
    print("Per-class AP (all sizes):")
    print(f"  {'class':<20s} {'#GT':>8s} {'AP50':>8s} {'AP50-95':>9s}")
    for r in per_class:
        a50 = f"{r['AP50']:.2f}" if np.isfinite(r["AP50"]) else "n/a"
        a95 = f"{r['AP50_95']:.2f}" if np.isfinite(r["AP50_95"]) else "n/a"
        print(f"  {r['class']:<20s} {r['n_gt']:>8d} {a50:>8s} {a95:>9s}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", default=None, help="one or more .pt weights")
    p.add_argument("--tags", nargs="*", default=None, help="optional names, one per --weights entry")
    p.add_argument("--manifest", default=None,
                   help="text file with 'name: /path/to/best.pt' lines (e.g. Table II best.txt)")
    p.add_argument("--exclude", default="", help="comma-separated manifest names to skip")
    p.add_argument("--dry-run", action="store_true", help="print the resolved model list and exit")
    p.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml", help="dataset YAML")
    p.add_argument("--split", default="val", help="dataset split key (val/test/train)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001, help="low threshold for AP evaluation")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300,
                   help="max detections per image (300 matches Ultralytics val; raise for crowded scenes)")
    p.add_argument("--area-space", choices=["original", "input", "both"], default="both",
                   help="measure object size in original pixels, at model input scale, or both")
    p.add_argument("--batch", type=int, default=1,
                   help="inference batch size (keep 1 for Mamba/SS2D models; results are unaffected)")
    p.add_argument("--fp32", action="store_true",
                   help="disable FP16 inference (default is FP16 on GPU, which Mamba models need to fit in 16GB)")
    p.add_argument("--device", default="", help="'' (auto), '0', '0,1', or 'cpu'")
    p.add_argument("--out", default="runs/ap_breakdown", help="output directory")
    p.add_argument("--limit", type=int, default=0,
                   help="legacy run_inference only; run_validation always uses the full split")
    p.add_argument("--save-json", action="store_true", help="also dump the COCO ground-truth JSON")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve model list: manifest + explicit --weights ----------------
    manifest_weights, manifest_tags = [], []
    if args.manifest:
        mpath = Path(args.manifest).expanduser().resolve()
        if not mpath.is_file():
            raise SystemExit(f"manifest not found: {mpath}")
        exclude = {x.strip().lower() for x in args.exclude.split(",") if x.strip()}
        for line in mpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            name, path = line.split(":", 1)
            name, path = name.strip(), path.strip()
            if name.lower() in exclude:
                continue
            manifest_weights.append(path)
            manifest_tags.append(name)

    weights_list = manifest_weights + list(args.weights or [])
    tags_list = list(manifest_tags)
    if args.weights:
        if args.tags:
            if len(args.tags) != len(args.weights):
                raise SystemExit("--tags must have the same length as --weights")
            tags_list += list(args.tags)
        else:
            tags_list += [Path(w).stem for w in args.weights]

    if not weights_list:
        raise SystemExit("provide --weights and/or --manifest")

    if args.dry_run:
        print(f"[dry-run] {len(weights_list)} models:")
        for name, w in zip(tags_list, weights_list):
            print(f"  {name:14s} exists={Path(w).is_file()!s:5s} {w}")
        return

    root, images, names = resolve_dataset(args.data, args.split)
    if args.limit:
        images = images[: args.limit]
    print(f"[info] dataset root: {root}")
    print(f"[info] split '{args.split}': {len(images)} images, {len(names)} classes")

    gt, path_to_id = build_gt(images, names, args.imgsz)
    n_gt = len(gt["annotations"])
    print(f"[info] GT: {len(gt['images'])} images, {n_gt} boxes")
    spaces = ["original", "input"] if args.area_space == "both" else [args.area_space]

    all_class_rows, all_size_rows, all_class_size_rows = [], [], []
    failed = []

    for i, weights in enumerate(weights_list):
        tag = tags_list[i].replace(os.sep, "_")
        print(f"\n[info] evaluating {tag}: {weights}")

        try:
            captured, official_map50, official_map = run_validation(weights, args, out_dir, tag)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {tag}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append((tag, f"{type(exc).__name__}: {exc}"))
            continue

        n_det = sum(len(c["pred_conf"]) for c in captured)
        print(f"[info] official val: mAP50={official_map50:.2f} mAP50-95={official_map:.2f}; "
              f"{n_det} detections")

        meta = {
            "weights": str(Path(weights).resolve()),
            "data": str(Path(args.data).resolve()),
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "fp16": (not args.fp32) and str(args.device).lower() != "cpu",
            "n_images": len(gt["images"]),
            "n_gt": n_gt,
            "n_det": n_det,
            "official_map50": official_map50,
            "official_map": official_map,
            "tag": tag,
        }

        per_class_main, size_tables, class_size_tables = evaluate_ultralytics(captured, names)

        for r in per_class_main:
            all_class_rows.append({"model": tag, **r})
        write_csv(out_dir / f"{tag}_per_class.csv", per_class_main)

        for space in spaces:
            for r in size_tables[space]:
                all_size_rows.append({"model": tag, "size_space": space, **r})
            for r in class_size_tables[space]:
                all_class_size_rows.append({"model": tag, "size_space": space, **r})
            write_csv(out_dir / f"{tag}_per_size_{space}.csv", size_tables[space])
            write_csv(out_dir / f"{tag}_per_class_size_{space}.csv", class_size_tables[space])

        write_markdown(out_dir / f"{tag}_summary.md", tag, per_class_main, size_tables, meta)
        (out_dir / f"{tag}_meta.json").write_text(
            json.dumps({**meta, "area_space": spaces}, indent=2), encoding="utf-8"
        )

        if args.save_json:
            (out_dir / f"{tag}_gt.json").write_text(json.dumps(gt), encoding="utf-8")

        print_summary(tag, per_class_main, size_tables)

        # free GPU memory before the next model (SS2D activations are large)
        del captured
        gc.collect()
        torch.cuda.empty_cache()

    # combined tables
    write_csv(out_dir / "summary_per_class.csv", all_class_rows)
    write_csv(out_dir / "summary_per_size.csv", all_size_rows)
    write_csv(out_dir / "summary_per_class_size.csv", all_class_size_rows)
    print(f"\n[done] outputs written to {out_dir}")
    print("  summary_per_class.csv / summary_per_size.csv / summary_per_class_size.csv")
    if failed:
        print(f"\n[warn] {len(failed)} model(s) failed:")
        for tag, err in failed:
            print(f"  {tag}: {err}")


if __name__ == "__main__":
    main()
