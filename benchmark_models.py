#!/usr/bin/env python3
"""
benchmark_models.py -- 14-model Benchmark Script
Uses YOLO built-in benchmark API (format="-", PyTorch only) to test inference speed + accuracy.

Usage:
    python benchmark_models.py                    # All models, using VisDrone validation set
    python benchmark_models.py --models yolo12n   # Only specified model
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath('.') + "/"

DEFAULT_BEST_TXT = ROOT + "best.txt"
DEFAULT_OUTPUT = ROOT + "benchmark_output"
DEFAULT_DATA = ROOT + "ultralytics/cfg/datasets/VisDrone.yaml"


def parse_opt():
    parser = argparse.ArgumentParser(description="14-model Benchmark - YOLO built-in benchmark API")
    parser.add_argument("--best-txt", type=str, default=DEFAULT_BEST_TXT,
                        help="path to best.txt")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="dataset yaml path (default VisDrone.yaml)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0,1")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def parse_best_txt(best_txt_path):
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


def load_model(model_path, device):
    from ultralytics import YOLO, RTDETR
    try:
        model = YOLO(model_path)
        return model, "YOLO"
    except Exception:
        pass
    model = RTDETR(model_path)
    return model, "RTDETR"


def count_params(model):
    try:
        return sum(p.numel() for p in model.model.parameters())
    except Exception:
        return 0


def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    else:
        return str(n)


def main():
    opt = parse_opt()

    if not os.path.isfile(opt.best_txt):
        print(f"[Error] best.txt not found: {opt.best_txt}")
        sys.exit(1)
    if not os.path.isfile(opt.data):
        print(f"[Error] dataset yaml not found: {opt.data}")
        sys.exit(1)

    all_models = parse_best_txt(opt.best_txt)
    if opt.models:
        all_models = [(n, p) for n, p in all_models if n in opt.models]
        if not all_models:
            print(f"[Error] specified models not found: {opt.models}")
            sys.exit(1)

    print(f"\n{'='*90}")
    print(f"  Mamba-YOLO Model Benchmark")
    print(f"  Dataset: {opt.data}")
    print(f"  Total models: {len(all_models)}")
    print(f"  API: model.benchmark(format='-')")
    print(f"{'='*90}\n")

    os.makedirs(opt.output, exist_ok=True)
    benchmark_rows = []

    for idx, (name, weight_path) in enumerate(all_models):
        print(f"\n{'─'*90}")
        print(f"  [{idx+1}/{len(all_models)}] {name}")
        print(f"  Weights: {weight_path}")
        print(f"{'─'*90}")

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

        # YOLO built-in benchmark - PyTorch format only
        # Set rect=False, benchmark's internal val() reads from model.args
        model.args["rect"] = False
        try:
            df = model.benchmark(
                data=opt.data,
                imgsz=opt.imgsz,
                device=opt.device,
                half=opt.half,
                format="-",       # PyTorch only, no export
                verbose=False,
            )
        except Exception as e:
            print(f"  [FAIL] benchmark failed: {e}")
            del model; import torch; torch.cuda.empty_cache()
            continue

        # Parse polars DataFrame
        import polars as pl
        if isinstance(df, pl.DataFrame) and len(df) > 0:
            row = df.row(0, named=True)
            cols = list(row.keys())
            metric_key = cols[3]
            raw_time = row.get("Inference time (ms/im)", 0)
            raw_fps = row.get("FPS", 0)
            inf_time = float(raw_time) if str(raw_time).replace(".", "").replace("-", "").isdigit() else 0.0
            fps_val = float(raw_fps) if str(raw_fps).replace(".", "").replace("-", "").isdigit() else 0.0
            metric_val = row.get(metric_key, "-")
            print(f"  Benchmark: PyTorch | {metric_key}={metric_val}")
            print(f"  Inference time: {inf_time:.2f} ms/im | FPS: {fps_val:.2f}")
        else:
            print(f"  [Warn] abnormal benchmark result")
            del model; import torch; torch.cuda.empty_cache()
            continue

        benchmark_rows.append({
            "name": name, "type": model_type,
            "params": format_params(params_n),
            "inf_time_ms": inf_time, "fps": fps_val,
        })

        del model; import torch; torch.cuda.empty_cache()

    # Summary table
    if benchmark_rows:
        sorted_rows = sorted(benchmark_rows, key=lambda x: x["inf_time_ms"])
        print(f"\n\n{'='*90}")
        print(f"  Benchmark Summary (sorted by inference time)")
        print(f"{'='*90}")
        print(f"  {'Rank':>4}  {'Model':<20} {'Type':>8} {'Params':>8} "
              f"{'Inference(ms)':>14} {'FPS':>8}")
        print(f"  {'─'*90}")
        for rank, r in enumerate(sorted_rows, 1):
            print(f"  {rank:>4}  {r['name']:<20} {r['type']:>8} {r['params']:>8} "
                  f"{r['inf_time_ms']:>13.2f}  {r['fps']:>7.2f}")
        print(f"{'='*90}\n")
        print(f"Done! {len(sorted_rows)}/{len(all_models)} models passed\n")
    else:
        print("\nNo benchmark results available\n")


if __name__ == "__main__":
    main()
