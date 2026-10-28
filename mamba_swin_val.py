from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/VisDrone.yaml', help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='output_dir/visdrone/ms-yolo-m/weights/best.pt', help='model weight path')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-swin/yolo12m-mamba-swin.yaml', help='model yaml path (only used when --model is empty)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optimizer', default='AdamW', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default='output_dir/visdrone', help='save to project/name')
    parser.add_argument('--name', default='mamba-swin-m', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--seed', type=int, default=0, help='global RNG seed')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "data": ROOT + opt.data,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "device": opt.device,
        "project": ROOT + opt.project,
        "name": opt.name,
        "seed": opt.seed
    }
    if opt.model:
        model_path = opt.model
        if not os.path.isabs(model_path):
            model_path = ROOT + model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'model weights not found: {model_path}')
    else:
        model_path = ROOT + opt.config
    print(f'val model: {model_path}')
    YOLO(model_path).val(**args)