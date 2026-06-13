#!/usr/bin/env python3
"""
test_image.py -- Single-image inference testing script
By default runs inference + latency test on all models in best.txt.
Supports manually specifying a single model.

Usage:
    python test_image.py                                             # All 14 models
    python test_image.py --model output_dir/visdrone/...             # Single model
    python test_image.py --models yolo12n mambayolo-B rt-detr-l     # Multiple models
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
import torch

ROOT = os.path.abspath('.') + "/"

# VisDrone 10 class names
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

DEFAULT_SOURCE = ROOT + "datasets/VisDrone/images/train2017/0000288_00201_d_0000786.jpg"
DEFAULT_OUTPUT = ROOT + "test_output"
DEFAULT_BEST_TXT = ROOT + "compare.txt"
WARMUP_RUNS = 3
BENCH_RUNS = 10


def parse_opt():
    parser = argparse.ArgumentParser(description="Single-image inference test - Mamba-YOLO models")
    parser.add_argument("--model", type=str, default=None,
                        help="model weights .pt path (if not set, runs all models in best.txt)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="list of model names, e.g. yolo12n mambayolo-B (default: all)")
    parser.add_argument("--best-txt", type=str, default=DEFAULT_BEST_TXT,
                        help="path to best.txt")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="path to test image")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="inference image size (default 640)")
    parser.add_argument("--device", type=str, default="0,1",
                        help="CUDA device, e.g. 0 or cpu")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="output directory (default ./test_output/)")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    return parser.parse_args()


def parse_best_txt(best_txt_path: str) -> list[tuple[str, str]]:
    """Parse best.txt, return [(model_name, weight_path), ...] list."""
    models = []
    with open(best_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            models.append((parts[0].strip(), parts[1].strip()))
    return models


def load_model(model_path: str, device: str):
    """Auto-detect model type and load. Try YOLO first, fallback to RTDETR."""
    from ultralytics import YOLO, RTDETR

    print(f"  [Loading] Attempting to load as YOLO format...")
    try:
        model = YOLO(model_path)
        print(f"  [OK] Loaded successfully, model type: YOLO")
        return model, "YOLO"
    except Exception as e_yolo:
        print(f"  [Retry] YOLO loading failed: {e_yolo}")

    print(f"  [Loading] Attempting to load as RTDETR format...")
    try:
        model = RTDETR(model_path)
        print(f"  [OK] Loaded successfully, model type: RTDETR")
        return model, "RTDETR"
    except Exception as e_rtdetr:
        print(f"  [Error] RTDETR loading also failed: {e_rtdetr}")
        raise RuntimeError(f"Unable to load model {model_path}")


def count_params(model) -> int:
    """Count model parameters."""
    try:
        return sum(p.numel() for p in model.model.parameters())
    except Exception:
        return 0


def format_params(n: int) -> str:
    """Format parameter count."""
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    else:
        return str(n)


def run_single_model(model, model_name, source, conf, imgsz, device, half, output_dir):
    """Warmup -> inference -> print results -> save image, return result dict."""
    # Warmup
    print(f"  [Warmup] Warming up {WARMUP_RUNS} runs...")
    for _ in range(WARMUP_RUNS):
        _ = model.predict(source=source, conf=conf, imgsz=imgsz,
                          device=device, half=half, rect=False, verbose=False)

    # Timed inference
    print(f"  [Bench] Running {BENCH_RUNS} timed inferences...")
    times = []
    last_results = None
    for i in range(BENCH_RUNS):
        t0 = time.perf_counter()
        results = model.predict(source=source, conf=conf, imgsz=imgsz,
                                device=device, half=half, rect=False, verbose=False)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)
        last_results = results
        print(f"    run {i+1}/{BENCH_RUNS}: {elapsed_ms:.2f} ms")

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / avg_ms
    print(f"  [Result] Avg: {avg_ms:.2f} +/- {std_ms:.2f} ms | FPS: {fps:.2f}")

    boxes = last_results[0].boxes
    num_dets = len(boxes) if boxes is not None else 0

    print_detections(last_results, VISDRONE_CLASSES, prefix="  ")

    source_name = Path(source).name
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{source_name}")
    last_results[0].save(filename=out_path)
    print(f"  [Save] Annotated image -> {out_path}")

    return {"name": model_name, "latency_ms": avg_ms, "latency_std_ms": std_ms,
            "fps": fps, "detections": num_dets}


def print_detections(results, class_names, prefix=""):
    """Print detection results."""
    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print(f"{prefix}[Result] No objects detected")
        return
    print(f"{prefix}{'='*66}")
    print(f"{prefix}Detected {len(boxes)} objects:")
    print(f"{prefix}{'='*66}")
    print(f"{prefix}{'#':>3}  {'Class':<18} {'Confidence':>8}  BBox (x1, y1, x2, y2)")
    print(f"{prefix}{'-'*66}")
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        conf_val = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"{prefix}{i+1:>3}  {cls_name:<18} {conf_val:>7.4f}  "
              f"[{xyxy[0]:>7.1f}, {xyxy[1]:>7.1f}, {xyxy[2]:>7.1f}, {xyxy[3]:>7.1f}]")
    print(f"{prefix}{'='*66}\n")


def main():
    opt = parse_opt()

    if not os.path.isfile(opt.source):
        print(f"[Error] Image not found: {opt.source}")
        sys.exit(1)

    source_name = Path(opt.source).name

    # Determine model list
    if opt.model:
        model_list = [("custom", opt.model)]
    else:
        if not os.path.isfile(opt.best_txt):
            print(f"[Error] best.txt not found: {opt.best_txt}")
            sys.exit(1)
        all_models = parse_best_txt(opt.best_txt)
        if opt.models:
            model_list = [(n, p) for n, p in all_models if n in opt.models]
            if not model_list:
                print(f"[Error] Specified models not found: {opt.models}")
                sys.exit(1)
        else:
            model_list = all_models

    print(f"\n{'='*80}")
    print(f"  Mamba-YOLO Single Image Inference Test")
    print(f"  Test image: {source_name}")
    print(f"  Model count: {len(model_list)}")
    print(f"  Warmup: {WARMUP_RUNS} runs | Timed: {BENCH_RUNS} runs")
    print(f"{'='*80}\n")

    os.makedirs(opt.output, exist_ok=True)
    results_summary = []

    for idx, (name, weight_path) in enumerate(model_list):
        print(f"\n{'─'*80}")
        print(f"  [{idx+1}/{len(model_list)}] {name}")
        print(f"  Weights: {weight_path}")
        print(f"{'─'*80}")

        if not os.path.isfile(weight_path):
            print(f"  [SKIP] weight file not found")
            continue

        try:
            model, model_type = load_model(weight_path, opt.device)
            params_n = count_params(model)
            print(f"  Type: {model_type} | Params: {format_params(params_n)}")
        except Exception as e:
            print(f"  [FAIL] model loading failed: {e}")
            continue

        try:
            result = run_single_model(model, name, opt.source, opt.conf, opt.imgsz,
                                      opt.device, opt.half, opt.output)
            if result:
                result["type"] = model_type
                result["params"] = format_params(params_n)
                results_summary.append(result)
        except Exception as e:
            print(f"  [FAIL] inference failed: {e}")

        del model
        torch.cuda.empty_cache()

    # Summary table
    if results_summary:
        sorted_results = sorted(results_summary, key=lambda x: x["latency_ms"])
        print(f"\n\n{'='*80}")
        print(f"  Test Summary (sorted by latency)")
        print(f"{'='*80}")
        print(f"  {'Rank':>4}  {'Model':<20} {'Type':>8} {'Params':>8} "
              f"{'Latency(ms)':>10} {'FPS':>8} {'Detections':>6}")
        print(f"  {'─'*80}")
        for rank, r in enumerate(sorted_results, 1):
            print(f"  {rank:>4}  {r['name']:<20} {r['type']:>8} {r['params']:>8} "
                  f"{r['latency_ms']:>9.2f}  {r['fps']:>7.2f}  {r['detections']:>5}")
        print(f"{'='*80}\n")

    print(f"Results saved to: {opt.output}")
    print(f"Done! {len(results_summary)}/{len(model_list)} models tested successfully\n")


if __name__ == "__main__":
    main()
