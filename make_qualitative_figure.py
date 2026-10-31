#!/usr/bin/env python3
"""Build the qualitative comparison figure (Original + 3 detectors).

Usage:
    python make_qualitative_figure.py
    python make_qualitative_figure.py --images runs/qualitative_random/selected_images.txt \
        --out-dir runs/qualitative_random --device 0
"""

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

TAG_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)

MODELS = [
    ("YOLOv12-M", "/public/zhongwenrui/Mamba-YOLO/output_dir/visdrone/yolo12-addLayer-m-/weights/best.pt", "#1f77b4"),
    ("MambaYOLO-B", "/public/zhongwenrui/Mamba-YOLO/output_dir/visdrone/mamba-yolo-b/weights/best.pt", "#2ca02c"),
    ("MS-YOLO-M (Ours)", "/dl/ms-yolo/runs/repaired_weights/mamba-swin-s3-repaired.pt", "#d62728"),
]
GT_COLOR = "#ffd400"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="runs/qualitative_random/selected_images.txt")
    ap.add_argument("--stats", default="runs/qualitative_random/selected_samples.csv")
    ap.add_argument("--out-dir", default="runs/qualitative_random")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--panel-w", type=int, default=1360,
                    help="panel width in pixels (default: 1360, native width of most val images)")
    ap.add_argument("--panel-h", type=int, default=0,
                    help="panel height in pixels; 0 = derive from each image's aspect ratio")
    ap.add_argument("--cell-w-in", type=float, default=3.2,
                    help="physical cell width in inches; DPI = panel_w / cell_w_in")
    ap.add_argument("--zoom", choices=["none", "inset"], default="none",
                    help="none: full images only (default); inset: add a zoom crop")
    ap.add_argument("--zoom-frac", type=float, default=0.26,
                    help="inset width as a fraction of the panel width")
    ap.add_argument("--dpi", type=int, default=0,
                    help="output DPI; 0 = auto (panel_w / cell_w_in)")
    return ap.parse_args()


def label_path_for(img_path: Path) -> Path:
    parts = list(img_path.parts)
    try:
        i = parts.index("images")
    except ValueError:
        return img_path.with_suffix(".txt")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_gt(img_path: Path, W: int, H: int):
    """Return GT boxes in original-image pixel coords."""
    lp = label_path_for(img_path)
    boxes = []
    if not lp.is_file():
        return boxes
    with open(lp) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def densest_window(boxes, W, H, win):
    """Pick the window of size `win` containing the most GT box centres."""
    if not boxes:
        x = (W - win) // 2
        y = (H - win) // 2
        return x, y
    cx = np.array([(b[0] + b[2]) / 2 for b in boxes])
    cy = np.array([(b[1] + b[3]) / 2 for b in boxes])
    stride = max(16, win // 10)
    best = (-1, (W - win) // 2, (H - win) // 2)
    for y in range(0, max(1, H - win + 1), stride):
        for x in range(0, max(1, W - win + 1), stride):
            n = int(((cx >= x) & (cx < x + win) & (cy >= y) & (cy < y + win)).sum())
            if n > best[0]:
                best = (n, x, y)
    return best[1], best[2]


def draw_full(img: np.ndarray, boxes, color, tag=None):
    """Draw boxes on a full-resolution copy and return a PIL image."""
    H, W = img.shape[:2]
    width = max(2, int(min(W, H) / 400))
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if tag:
        bbox = draw.textbbox((6, 6), tag, font=TAG_FONT)
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=(0, 0, 0))
        draw.text((6, 6), tag, fill=color, font=TAG_FONT)
    return pil


def render_panel(img: np.ndarray, boxes, color, panel_w, panel_h, zoom_rect=None,
                 box_width=3, tag=None):
    """Resize image to the panel size, draw boxes, return PIL image."""
    H, W = img.shape[:2]
    scale_x, scale_y = panel_w / W, panel_h / H
    pil = Image.fromarray(img).resize((panel_w, panel_h), Image.LANCZOS)
    draw = ImageDraw.Draw(pil)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                       outline=color, width=box_width)
    if zoom_rect is not None:
        zx, zy, zw, zh = zoom_rect
        draw.rectangle([zx * scale_x, zy * scale_y, (zx + zw) * scale_x, (zy + zh) * scale_y],
                       outline=(255, 255, 255), width=2)
    if tag:
        bbox = draw.textbbox((4, 3), tag, font=TAG_FONT)
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=(0, 0, 0))
        draw.text((4, 3), tag, fill=color, font=TAG_FONT)
    return pil


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(p.strip()) for p in Path(args.images).read_text().splitlines() if p.strip()]
    if not image_paths:
        raise SystemExit(f"No images listed in {args.images}")

    stats = {}
    if Path(args.stats).is_file():
        with open(args.stats, newline="") as f:
            for r in csv.DictReader(f):
                stats[r["stem"]] = r

    # panel heights follow each image's aspect ratio (no distortion, no upscaling)
    sizes = []
    for p in image_paths:
        with Image.open(p) as im:
            sizes.append(im.size)  # (W, H)
    panel_hs = [
        args.panel_h if args.panel_h else int(round(args.panel_w * H / W))
        for W, H in sizes
    ]

    # ---- load detectors once ----
    from ultralytics import YOLO
    loaded = []
    for name, path, color in MODELS:
        if not Path(path).is_file():
            raise SystemExit(f"[MISS] model weight not found: {path}")
        print(f"[Load] {name}: {path}")
        loaded.append((name, YOLO(path), color))

    panels = []  # rows: list of rendered PIL panels
    zoom_rects = []
    orig_dir = out_dir / "originals"
    panel_dir = out_dir / "panels_fullres"
    orig_dir.mkdir(exist_ok=True)
    panel_dir.mkdir(exist_ok=True)

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        img = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        gt = read_gt(img_path, W, H)

        # keep a pristine copy of the original image for manual zooming
        shutil.copy2(img_path, orig_dir / img_path.name)

        if args.zoom == "inset":
            win = int(min(W, H) * 0.38)
            zx, zy = densest_window(gt, W, H, win)
            zoom = (zx, zy, win, win)
        else:
            zoom = None
        zoom_rects.append(zoom)

        ph = panel_hs[idx - 1]
        row = [render_panel(img, gt, GT_COLOR, args.panel_w, ph, zoom,
                            tag="GT" if gt else None)]
        draw_full(img, gt, GT_COLOR, tag="GT" if gt else None).save(
            panel_dir / f"{stem}__GT.jpg", quality=95)

        for name, model, color in loaded:
            res = model.predict(source=str(img_path), imgsz=args.imgsz,
                                conf=args.conf, iou=args.iou,
                                device=args.device, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            row.append(render_panel(img, boxes, color, args.panel_w, ph, zoom))
            safe = name.split(" (")[0].replace(" ", "-")
            draw_full(img, boxes, color, tag=safe).save(
                panel_dir / f"{stem}__{safe}.jpg", quality=95)
            print(f"  [Pred] image {idx} {name}: {len(boxes)} detections")
        panels.append(row)

    # ---- compose with matplotlib ----
    n_rows, n_cols = len(panels), 1 + len(loaded)
    cell_w_in = args.cell_w_in
    fig_w = cell_w_in * n_cols
    fig_h = sum(cell_w_in * ph / args.panel_w for ph in panel_hs)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=panel_hs)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)]
                     for i in range(n_rows)])
    fig.subplots_adjust(left=0.022, right=1.0, bottom=0.010, top=0.975,
                        wspace=0.012, hspace=0.035)
    titles = ["Original (GT)"] + [m[0] for m in loaded]

    for i, row in enumerate(panels):
        stem = image_paths[i].stem
        st = stats.get(stem, {})
        row_label = f"Image {i+1}"
        for j, panel in enumerate(row):
            ax = axes[i, j]
            ax.imshow(panel, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#bbbbbb"); spine.set_linewidth(0.6)
            if i == 0:
                ax.set_title(titles[j], fontsize=10.5, fontweight="bold", pad=4,
                             color=("#d62728" if "Ours" in titles[j] else "#222222"))
            if j == 0:
                ax.set_ylabel(row_label, fontsize=8, rotation=90, labelpad=6)

            # optional zoom inset (same GT-driven region in every column)
            if args.zoom == "inset" and zoom_rects[i] is not None:
                zx, zy, zw, zh = zoom_rects[i]
                W, H = Image.open(image_paths[i]).size
                sx, sy = args.panel_w / W, panel_hs[i] / H
                aspect = (zh * sy) / (zw * sx)
                inset_w = args.zoom_frac
                inset_h = inset_w * args.panel_w / panel_hs[i] * aspect
                axin = ax.inset_axes([1 - inset_w - 0.015, 0.015, inset_w, inset_h])
                crop = np.asarray(panel.crop((int(zx * sx), int(zy * sy),
                                              int((zx + zw) * sx), int((zy + zh) * sy))))
                axin.imshow(crop)
                axin.set_xticks([]); axin.set_yticks([])
                for spine in axin.spines.values():
                    spine.set_edgecolor("#ffffff"); spine.set_linewidth(1.2)
                axin.patch.set_alpha(1.0)

    dpi = args.dpi if args.dpi > 0 else args.panel_w / args.cell_w_in
    print(f"[Info] panel={args.panel_w}px wide, cell={cell_w_in}in, output DPI={dpi:.1f}")
    fig.savefig(out_dir / "figure_qualitative.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "figure_qualitative.pdf", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- caption + protocol snippet ----
    sample_rows = []
    for i, p in enumerate(image_paths, 1):
        st = stats.get(p.stem, {})
        if st:
            sample_rows.append(
                f"| {i} | `{p.name}` | {st['n_objects']} | {float(st['mean_size_px']):.1f} |")
    sample_table = "\n".join(sample_rows)

    zoom_text = (
        " White rectangles and the bottom-right insets show the same GT-driven "
        "region (the densest 38% window) at higher magnification."
        if args.zoom == "inset" else ""
    )
    caption = f"""# Figure caption (draft)

**Qualitative comparison on the VisDrone2019 validation set.** The four rows show
images drawn by simple random sampling with a fixed seed (2026) from the 521
validation images that contain at least 10 annotated objects
(each image had probability {len(image_paths)}/521 of being selected); no manual
selection was performed. Columns show the original image with ground-truth boxes
(yellow), YOLOv12-M, MambaYOLO-B, and MS-YOLO-M. All detectors use identical
settings: input 640x640, confidence 0.25, NMS IoU 0.7, FP32.{zoom_text}

Sampled images:

| Row | File | Objects | Mean size (px) |
|---|---|---:|---:|
{sample_table}
"""
    (out_dir / "figure_caption.md").write_text(caption)
    (out_dir / "figure_caption.md").write_text(caption)
    print(f"\n[OK] figure    -> {out_dir}/figure_qualitative.png")
    print(f"[OK] figure    -> {out_dir}/figure_qualitative.pdf")
    print(f"[OK] originals -> {orig_dir}/  ({len(image_paths)} raw images)")
    print(f"[OK] full-res  -> {panel_dir}/  ({len(image_paths) * (1 + len(loaded))} annotated images)")
    print(f"[OK] caption   -> {out_dir}/figure_caption.md")


if __name__ == "__main__":
    main()
