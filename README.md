# MS-YOLO: A Mamba–Swin Transformer Hybrid Architecture for Parameter- and Computation-Efficient Small Object Detection

![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![PyTorch 2.3.0](https://img.shields.io/badge/pytorch-2.3.0-blue.svg) ![License MIT](https://img.shields.io/badge/license-MIT-green) [![Checkpoints](https://img.shields.io/badge/checkpoints-GitHub%20Releases-orange)](https://github.com/zhongwr2/ms-yolo/releases)

MS-YOLO is a scale-adaptive **Mamba–Swin hybrid detector** for small object detection in UAV imagery. Mamba state-space blocks are placed in the shallow, high-resolution stages (global context at linear cost), and a lightweight Swin Transformer block is placed in the deep stage (efficient local refinement). The detector further uses a high-resolution **P2 detection branch**, a **Simplified FPN** (top-down only, without the PAN pathway and the P5 branch), and a parameter-neutral patch embedding (**SOPE**).

- **MS-YOLO-M**: **47.6 % mAP50** on VisDrone2019 val with **10.62 M** parameters and **37.1 GFLOPs** (mean of 3 seeds)
- **MS-YOLO-L**: **50.6 % mAP50**, beating YOLOv12-X (45.7 %) with about one third of its parameters
- **TinyPerson**: 27.6 % mAP50 versus 19.9 % for YOLOv12-M
- Parameter- and computation-efficient rather than latency-optimal: 1.6–2.4× slower than YOLOv12 at the same scale (PyTorch, NVIDIA A10, no TensorRT)

All pretrained checkpoints are available in [GitHub Releases](#checkpoints).

## News

- **2026-10**: all pretrained checkpoints (VisDrone2019 and TinyPerson) are released on GitHub Releases.

## Results on VisDrone2019-DET (val)

![VisDrone Results](./images/table2.png)

| Model | Params (M) | GFLOPs | mAP50 (%) | mAP50:95 (%) |
|---|---:|---:|---:|---:|
| MS-YOLO-N | 0.97 | 5.8 | 32.25 | 18.55 |
| MS-YOLO-S | 2.98 | 12.5 | 41.74 | 25.00 |
| MS-YOLO-M (3 seeds) | 10.62 | 37.1 | **47.62 ± 0.54** | **29.16 ± 0.35** |
| MS-YOLO-L | 20.14 | 118.5 | 50.56 | 30.93 |

All models are trained from scratch for 150 epochs at 640×640 (AdamW, no ImageNet pre-training). Latency and FPS in the paper are measured with PyTorch (no TensorRT), batch size 1, on an NVIDIA A10. MS-YOLO is parameter- and computation-efficient rather than latency-optimal; see the paper for the full comparison.

## Installation

MS-YOLO is developed with PyTorch 2.3.0 (CUDA 12.1 wheels) and CUDA 12.8, and is tested with Python 3.11. 

### Clone the repository 

```bash
git clone https://github.com/zhongwr2/ms-yolo.git
cd ms-yolo
```

### Create and activate a conda environment
```bash
conda create -n msyolo -y python=3.11
conda activate msyolo
```

### Install PyTorch

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### Install dependencies

```bash
pip install timm einops
cd selective_scan && pip install . --no-build-isolation
cd .. && pip install -v -e .
```

## Datasets

### VisDrone2019-DET

Make sure your dataset is structured as follows:
```
├── VisDrone
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
```
`VisDrone.yaml` contains the `visdrone2yolo` conversion (`instances_*.json` → YOLO `labels/`), which runs automatically on first use; if the dataset is not found, it is downloaded and converted to the structure above. Point `path` to your dataset root:

```bash
vi ultralytics/cfg/datasets/VisDrone.yaml   # e.g. path: /your/datasets/VisDrone
```

### TinyPerson

The TinyPerson experiments in the paper use the same YOLO-format pipeline (794 training / 816 test images, single class `person`):

```bash
vi ultralytics/cfg/datasets/TinyPerson.yaml   # e.g. path: /your/datasets/TinyPerson
```

## Training

```bash
python mamba_swin_train.py \
  --config ultralytics/cfg/models/mamba-swin/yolo12m-mamba-swin.yaml \
  --data   ultralytics/cfg/datasets/VisDrone.yaml \
  --epochs 150 --batch_size 8 --imgsz 640 --device 0,1 --seed 0
```

All models are trained from scratch for 150 epochs at 640×640 (AdamW, `lr0 = 0.01`, no ImageNet pre-training).

Available model configs (`ultralytics/cfg/models/mamba-swin/`):

| Config | Description |
|---|---|
| `yolo12-mamba-swin.yaml` | MS-YOLO: SOPE + P2 branch + Simplified FPN |
| `yolo12-mamba-swin-no-sope.yaml` | SimpleStem instead of SOPE |
| `yolo12-mamba-swin-only-p2.yaml` | P2 branch + PAFPN + SimpleStem |
| `yolo12-mamba-swin-p3.yaml` | No P2 branch ({P3, P4, P5}) + PAFPN + SimpleStem |
| `yolo12-mamba-swin-pafpn.yaml` | PAFPN instead of the Simplified FPN |
| `yolo12-mamba-swin-all-mamba.yaml` | All-Mamba backbone |
| `yolo12-mamba-swin-all-swin.yaml` | All-Swin backbone |
| `yolo12-mamba-swin-reversed.yaml` | Reversed allocation (Swin shallow, Mamba deep) |

## Evaluation and inference

```bash
# validation on a dataset split
python mamba_swin_val.py --model path/to/weights/best.pt \
    --data ultralytics/cfg/datasets/VisDrone.yaml --imgsz 640

# single-image inference + latency benchmark
python test_image.py --model path/to/weights/best.pt
```

```python
from ultralytics import YOLO

model = YOLO("path/to/weights/best.pt")
metrics = model.val(data="ultralytics/cfg/datasets/VisDrone.yaml", imgsz=640, split="val")
print(metrics.box.map50, metrics.box.map)
```

## Checkpoints

All pretrained checkpoints are hosted on GitHub Releases:

| Release | Contents |
|---|---|
| [`VisDrone2019`](https://github.com/zhongwr2/ms-yolo/releases/tag/VisDrone2019) | Main comparison, component/backbone ablations, P2 baselines |
| [`TinyPerson`](https://github.com/zhongwr2/ms-yolo/releases/tag/TinyPerson) | TinyPerson evaluation |

Each `.zip` is the training output of one run; unzip it and load `weights/best.pt` (if the archive contains a single `.pt` file, load that file directly). Table 2 and Table 3 of the paper are the main comparisons on VisDrone2019 and TinyPerson, Table 4 the component ablation, Table 5 the backbone ablation, and Table 7 the P2-attribution baselines.

Verify a download with:

```bash
sha256sum MS-YOLO-m-seed0.zip   # compare with the values below
```

### VisDrone2019 — main results and baselines

| Checkpoint | Experiment | Size |
|---|---|---:|
| [MS-YOLO-n.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/MS-YOLO-n.zip)<br><sub>`aa94086af2121bc5364664c3ad8269c3b8240e553ddf41831f92c4e0bf7415f7`</sub> | MS-YOLO-N (Table 2) | 12.2 MB |
| [MS-YOLO-s.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/MS-YOLO-s.zip)<br><sub>`a5090b46c251657f63c9f1f1d032f2be362b70191896981748778a4b5a536242`</sub> | MS-YOLO-S (Table 2) | 19.6 MB |
| [MS-YOLO-m-seed0.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/MS-YOLO-m-seed0.zip)<br><sub>`033d17164181dfa4c083ac44033f9318bf008a924e29fea3a47480906a27cbd5`</sub> | MS-YOLO-M, seed 0 (Table 2; latency and qualitative results) | 44.5 MB |
| [MS-YOLO-m-seed1.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/MS-YOLO-m-seed1.zip)<br><sub>`73c14419effd9c89fcbbaf47ca1ed212e9e98b70ff585ab51bc6ca021bcc967a`</sub> | MS-YOLO-M, seed 1 | 44.3 MB |
| [MS-YOLO-m-seed2.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/MS-YOLO-m-seed2.zip)<br><sub>`587256f09cc7ad85549f65e92ce0d246a3dc74af48b862f32958fc0d57e7a944`</sub> | MS-YOLO-M, seed 2 | 44.3 MB |
| [MS-YOLO-l.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/MS-YOLO-l.zip)<br><sub>`743ccfa200b4c49e91847acda058518ed910b7ce3d56f441722d2971a35b6ebc`</sub> | MS-YOLO-L (Table 2) | 116 MB |
| [yolo12n.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/yolo12n.zip)<br><sub>`9fcde1fbe16969e2be776816e41f359fd8fc83f022a16cc8ac4c6a1c90a4cfeb`</sub> | YOLOv12-N baseline | 17.9 MB |
| [yolo12s.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/yolo12s.zip)<br><sub>`c9c55c4680f68e1935586a98e24f09517a9e821d42ef6fbb7521f3d152b4533e`</sub> | YOLOv12-S baseline | 39.2 MB |
| [yolo12m.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/yolo12m.zip)<br><sub>`f298f62eb999278bf52aeaa6be3d682fa15481e8cd1e414619296d89d711eabf`</sub> | YOLOv12-M baseline | 77.3 MB |
| [yolo12l.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/yolo12l.zip)<br><sub>`b106f012e37255c97f354a4e9013191a5af33acee0a6c402a3ed6ce2b8082006`</sub> | YOLOv12-L baseline | 97.3 MB |
| [yolo12x.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/yolo12x.zip)<br><sub>`916005e62d436662da0c6806cfd1cb97f46b7fe6517fcf5d70075697da320970`</sub> | YOLOv12-X baseline | 211 MB |
| [mambayolo-T.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/mambayolo-T.zip)<br><sub>`6531d75079506e2b4d6c0dced971b3a195356b6c0da400c1de0b2d1d0a7e94ce`</sub> | MambaYOLO-T baseline | 29.8 MB |
| [mambayolo-B.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/mambayolo-B.zip)<br><sub>`4e33fb8b52c47a0f552827416a25bb8cc8796b1658155743b0e9087e54855cc2`</sub> | MambaYOLO-B baseline | 75.7 MB |
| [mambayolo-L.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/mambayolo-L.zip)<br><sub>`29d03e4cffab2ea8755a01ba8a10cc039243a5b8b1cd6b650977a5bfc571fb2e`</sub> | MambaYOLO-L baseline | 198 MB |

### VisDrone2019 — ablations

| Checkpoint | Experiment | Size |
|---|---|---:|
| [abl-PAFPN+simplestem.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-PAFPN+simplestem.zip)<br><sub>`70e6c0851625c963e7ccfbbcb513a5e34c167322333a151891e29c23e38e653e`</sub> | Table 4 (a): {P3, P4, P5} + PAFPN + SimpleStem | 41.6 MB |
| [abl-P2+PAFPN+simplestem.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-P2+PAFPN+simplestem.zip)<br><sub>`838e27a1c6f243cc3ed4be78cfc3b0d7d26ef8f877aba903d60efdc19b8a26e2`</sub> | Table 4 (b): P2 branch + PAFPN + SimpleStem | 44.2 MB |
| [abl-P2+FPN+simplestem.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-P2+FPN+simplestem.zip)<br><sub>`bafa6b8c43fc3fbec309ceeb05190fbc0c58848425de9be2ea517c0d8b041e8d`</sub> | Table 4 (c), seed 0: Simplified FPN + SimpleStem | 29 MB |
| [abl-msyolo-m-simplestem-seed1.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-msyolo-m-simplestem-seed1.zip)<br><sub>`3a76d54306930969bdd74fe7232dad4a812fe49e923dadde6328049296267f08`</sub> | Table 4 (c), seed 1 | 44.3 MB |
| [abl-msyolo-m-simplestem-seed2.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-msyolo-m-simplestem-seed2.zip)<br><sub>`836b3155ad00268d6f4533186d92d66d4476c8e3bd3cc15075da4c9bba33180b`</sub> | Table 4 (c), seed 2 | 44.4 MB |
| [abl-P2+PAFPN+SOPE.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-P2+PAFPN+SOPE.zip)<br><sub>`920567030582de8b9180f1438aaf8cf93df41ad39444d719b742e32566613c88`</sub> | Table 4 (d): P2 branch + PAFPN + SOPE | 27.8 MB |
| [abl-all-mamba-m-seed0.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-all-mamba-m-seed0.zip)<br><sub>`5e66162cc907a5dc2554d81e46021e311c5bfcb081dff8df8e22afe693cc19bb`</sub> | Table 5: all-Mamba, seed 0 | 66.4 MB |
| [abl-all-mamba-m-seed1.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-all-mamba-m-seed1.zip)<br><sub>`21125c5875698e428e1e9871dfac66478d274d46ccd7b74aecbfe098f3d57527`</sub> | Table 5: all-Mamba, seed 1 | 66.4 MB |
| [abl-all-mamba-m-seed2.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-all-mamba-m-seed2.zip)<br><sub>`967d2717928911b97e8d35fac4229435665c50a3bfafd3b1dc14ad8132263162`</sub> | Table 5: all-Mamba, seed 2 | 66.4 MB |
| [abl-all-swin-m.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-all-swin-m.zip)<br><sub>`518a40a4906e19f0236417486e992d6a47de43cf6f15575ca18797b35aeb8c3c`</sub> | Table 5: all-Swin | 46.4 MB |
| [abl-reversed-m.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-reversed-m.zip)<br><sub>`06483d6e1abed09f3b5d61f988f27e7e845ed65603dab77876bc957d8c9ccf3c`</sub> | Table 5: reversed allocation (Swin shallow, Mamba deep) | 38.7 MB |
| [abl-yolo12-p2-m.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-yolo12-p2-m.zip)<br><sub>`cafe224f8d9dd8c9d6ba0ebe3c263f8de3aa00a111b961357bb010771926b45c`</sub> | Table 7: YOLOv12-M + P2 | 147 MB |
| [abl-yolo12-p2-s.zip](https://github.com/zhongwr2/ms-yolo/releases/download/VisDrone2019/abl-yolo12-p2-s.zip)<br><sub>`f0ff0adbbf0361855d90d5423722e1926f7fdb3adf4f4edc970eb6d2077b9dc6`</sub> | Table 7: YOLOv12-S + P2 | 40.4 MB |

Table 4 (e), the full model, corresponds to `MS-YOLO-m-seed0/1/2.zip` listed above; the hybrid backbone row of Table 5 is the same set of runs.

### TinyPerson

| Checkpoint | Experiment | Size |
|---|---|---:|
| [ms-yolo-n.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/ms-yolo-n.zip)<br><sub>`4d1e02b53e956da717882c3de6b07a00f4241371edafea84dba234af0f7fcc9c`</sub> | MS-YOLO-N | 12.8 MB |
| [ms-yolo-s.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/ms-yolo-s.zip)<br><sub>`2d21b03b44b579cc1b874817709b225a42aceb42547cf8cb2affb5ddf1759591`</sub> | MS-YOLO-S | 27.1 MB |
| [ms-yolo-m-seed0.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/ms-yolo-m-seed0.zip)<br><sub>`f0516a83d6c7cf7c8c530cce74e4bac3f29b122c18e6b466342bb1f5ff849899`</sub> | MS-YOLO-M, seed 0 | 79.8 MB |
| [ms-yolo-m-seed1.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/ms-yolo-m-seed1.zip)<br><sub>`53bca99b54ead27e90785a109e91f7d99f19ec366c1bb6976595c7430acef65a`</sub> | MS-YOLO-M, seed 1 | 79.8 MB |
| [ms-yolo-m-seed2.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/ms-yolo-m-seed2.zip)<br><sub>`7ea7a065d4a1a8e6a13e74edb689b3148b86b6e0dbaa7a9e64ccd294796d27ca`</sub> | MS-YOLO-M, seed 2 | 79.8 MB |
| [ms-yolo-l.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/ms-yolo-l.zip)<br><sub>`3b9057e4a05eef3eff53e45e9873b92297ff92e09019255febafb52de5656f8b`</sub> | MS-YOLO-L | 145 MB |
| [yolo12-n.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/yolo12-n.zip)<br><sub>`8fe981d097140244ceb31768f437da943ec1049d4bd064b0e2e57369d1269dcd`</sub> | YOLOv12-N baseline | 24.3 MB |
| [yolo12-s.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/yolo12-s.zip)<br><sub>`2342495155d3f12709715a0fccb77a022b673f9cb011ece9b2040fd1a569c4a2`</sub> | YOLOv12-S baseline | 71.7 MB |
| [yolo12-m.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/yolo12-m.zip)<br><sub>`6ed1198a428030c7662c9c80d741790afa1899aca249ff899bd019823f073afc`</sub> | YOLOv12-M baseline | 149 MB |
| [yolo12-l.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/yolo12-l.zip)<br><sub>`aecf41a193736803bc92c57f02400fe028cfc34871db6c9a35efb6f069013e36`</sub> | YOLOv12-L baseline | 194 MB |
| [yolo12-x.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/yolo12-x.zip)<br><sub>`637bf485c7ac24cca03ffb116938f242c0288562c9dc899243443265daff32da`</sub> | YOLOv12-X baseline | 424 MB |
| [mambayolo-T.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/mambayolo-T.zip)<br><sub>`989966e33bfd52e4079098963654079a74b4781eaa57d0a5e91f79c38d5b0a80`</sub> | MambaYOLO-T baseline | 48.3 MB |
| [mambayolo-B.zip](https://github.com/zhongwr2/ms-yolo/releases/download/TinyPerson/mambayolo-B.zip)<br><sub>`da4f76cafcbd41ae7a786648590d68e58afca8195e83c6eac3d660721cd27272`</sub> | MambaYOLO-B baseline | 160 MB |

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhong2026msyolo,
  title   = {MS-YOLO: A Mamba--Swin Transformer Hybrid Architecture for
             Parameter- and Computation-Efficient Small Object Detection},
  author  = {Zhong, Wenrui and Qin, Chang and Pang, Guangyao and Feng, Meiyan},
  journal = {Image and Vision Computing},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgement

This repo is modified from the open-source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan implementation comes from [VMamba](https://github.com/MzeroMiko/VMamba) and [mamba_ssm](https://github.com/state-spaces/mamba).

## License

This project is released under the [MIT License](LICENSE).

