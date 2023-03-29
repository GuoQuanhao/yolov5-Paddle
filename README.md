<div align="center">

  <a href="https://github.com/GuoQuanhao/YOLOv5-Paddle" target="_blank">
    <img width="1024", src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/imgs/logo.png"></a>

<br>
  <a href="https://aistudio.baidu.com/aistudio/projectdetail/2580805?contributionType=1"><img width="100" src="https://raw.githubusercontent.com/GuoQuanhao/yolov5-Paddle/main/data/assets/AIStudio.png" alt="Run on AIstudio"></a>
<br>

[English](README.md) | [简体中文](README.zh-CN.md)

This <a href="https://github.com/GuoQuanhao/yolov5-Paddle">YOLOv5-Paddle</a> 🚀 notebook by <a href="https://github.com/GuoQuanhao">GuoQuanhao</a> presents simple train, validate and predict and export examples. YOLOv5-Paddle now supports conversion of single-precision and half-precision models in multiple formats. Contact me at <a href="https://github.com/GuoQuanhao">github</a> for professional support.
<br>
<br>&#9745; [PaddleLite](https://www.paddlepaddle.org.cn/lite/v2.12/api_reference/python_api/opt.html) &#9745; [PaddleInference](https://www.paddlepaddle.org.cn/inference/v2.4/guides/export_model/paddle_model_export.html) &#9745; [ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) &#9745; [OpenVIVO](https://github.com/openvinotoolkit/openvino) &#9745; [TensorRT](https://github.com/PaddlePaddle/Paddle2ONNX)<br>
</div>

## <div align="center">Detection</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PaddlePaddle>=2.4.0**](https://www.paddlepaddle.org.cn/).

```bash
git clone https://github.com/GuoQuanhao/yolov5-Paddle  # clone
cd yolov5-Paddle
pip install -r requirements.txt  # install
```

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/scripts/get_coco.sh)
results. [Models](https://github.com/GuoQuanhao/yolov5-Paddle/tree/main/models)
and [datasets](https://github.com/GuoQuanhao/yolov5-Paddle/tree/main/data) download automatically from the latest
YOLOv5 [release](https://github.com/GuoQuanhao/yolov5-Paddle/releases/tag/v2.0). Batch sizes shown for V100-16GB

```bash
# (from scratch)Single-GPU or CPU
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128  --device ''
                                                                 yolov5s                    64            cpu
                                                                 yolov5m                    40            0
                                                                 yolov5l                    24            1
                                                                 yolov5x                    16            2
															 
# (pretrained)Single-GPU or CPU
python train.py --data coco.yaml --epochs 300 --weights yolov5n.pdparams --batch-size 128  --device ''
                                                        yolov5s                       64            cpu
                                                        yolov5m                       40            0
                                                        yolov5l                       24            1
                                                        yolov5x                       16            2
```

```bash
# Multi-GPU, from scratch and pretrained as above
python -m paddle.distributed.launch --gpus 0,1,2,3 train.py --weights '' --cfg yolov5n.yaml --batch-size 128  --data coco.yaml --epochs 300 --device 0,1,2,3
                                                                                 yolov5s                    64
                                                                                 yolov5m                    40
                                                                                 yolov5l                    24
                                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">
</details>

<details>
<summary>Evaluation</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/scripts/get_coco.sh)
results. [Models](https://github.com/GuoQuanhao/yolov5-Paddle/tree/main/models)
and [datasets](https://github.com/GuoQuanhao/yolov5-Paddle/tree/main/data) download automatically from the latest
YOLOv5 [release](https://github.com/GuoQuanhao/yolov5-Paddle/releases/tag/v2.0). Batch sizes shown for V100-16GB

```bash
# (from scratch)Single-GPU or CPU
python val.py --data coco.yaml --weights yolov5n.pdparams --img 640 --conf 0.001 --iou 0.65 --device ''
                                         yolov5s                                                     cpu
                                         yolov5m                                                     0
                                         yolov5l                                                     1
                                         yolov5x                                                     2
```
</details>

<details>
<summary>Inference</summary>

YOLOv5 PaddlePaddle inference. Models download automatically from the latest

```python
# Model
python hubconf.py  # or yolov5n - yolov5x6, custom
```

`detect.py` runs inference on a variety of sources, downloading models automatically from
the Baidu Drive and saving results to `runs/detect`.

```bash
python detect.py --weights yolov5s.pdparams --source 0                         # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>benchmark</summary>

```bash
python benchmarks.py --weights ./yolov5s.pdparams --device 0
```
```
Benchmarks complete (187.81s)
            Format  Size (MB)  mAP50-95  Inference time (ms)
0     PaddlePaddle       13.9    0.4716                 9.75
1  PaddleInference       27.8    0.4716                20.82
2             ONNX       27.6    0.4717                32.23
3         TensorRT       32.2    0.4717                 3.05
4         OpenVINO       27.9    0.4717                43.67
5       PaddleLite       27.8    0.4717               264.86
```
</details>

<details>
  <summary>YOLOv5 and YOLOv5-P5 640 Figure</summary>
<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes</summary>

- **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
- **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
- **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pdparams yolov5s6.pdparams yolov5m6.pdparams yolov5l6.pdparams yolov5x6.pdparams`

</details>

<details>
 <summary>Detection Checkpoints</summary>

Accuracy, params and flops verificated by PaddlePaddle, speed is from original YOLOv5
| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| YOLOv5n              | 640                   | 28.0                 | 45.7              | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| YOLOv5s              | 640                   | 37.4                 | 56.8              | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| YOLOv5m              | 640                   | 45.3                 | 64.1              | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| YOLOv5l              | 640                   | 49.0                 | 67.4              | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| YOLOv5x              | 640                   | 50.6                 | 68.8              | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                 |                       |                      |                   |                              |                               |                                |                    |                        |
| YOLOv5n6            | 1280                  | 36.0                 | 54.4              | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| YOLOv5s6            | 1280                  | 44.8                 | 63.7              | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| YOLOv5m6            | 1280                  | 51.3                 | 69.3              | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| YOLOv5l6            | 1280                  | 53.7                 | 71.3              | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| YOLOv5x6<br>+ [TTA] | 1280<br>1536          | 55.0<br>**55.8**     | 72.7<br>**72.7**  | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |
</details>

<details>
  <summary>Table Notes</summary>

- All checkpoints are trained to 300 epochs with default settings. Nano and Small models use [hyp.scratch-low.yaml](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/hyps/hyp.scratch-low.yaml) hyps, all others use [hyp.scratch-high.yaml](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [Test Time Augmentation](https://github.com/ultralytics/yolov5/issues/303) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

<details>
<summary>Export</summary>

```bash
python export.py --weights yolov5n.pdparams --include paddleinfer onnx engine openvino paddlelite
						   yolov5s.pdparams
						   yolov5m.pdparams
						   yolov5l.pdparams
						   yolov5x.pdparams
```
You can use `--dynamic` or `--half` to get dynamic dimension or half-precision model.
</details>

## <div align="center">Segmentation</div>

<details>
  <summary>Segmentation Checkpoints</summary>

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/61612323/204180385-84f3aca9-a5e9-43d8-a617-dda7ca12e54a.png"></a>
</div>

| Model                                                                                      | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Train time<br><sup>300 epochs<br>A100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TRT A100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ------------------------------------------------------------------------------------------ | --------------------- | -------------------- | --------------------- | --------------------------------------------- | ------------------------------ | ------------------------------ | ------------------ | ---------------------- |
| YOLOv5n-seg | 640                   | 27.2                 | 23.5                  | 80:17                                         | **62.7**                       | **1.2**                        | **2.0**            | **7.1**                |
| YOLOv5s-seg | 640                   | 37.3                 | 31.8                  | 88:16                                         | 173.3                          | 1.4                            | 7.6                | 26.4                   |
| YOLOv5m-seg | 640                   | 44.7                 | 37.5                  | 108:36                                        | 427.0                          | 2.2                            | 22.0               | 70.8                   |
| YOLOv5l-seg | 640                   | 48.7                 | 40.3                  | 66:43 (2x)                                    | 857.4                          | 2.9                            | 47.9               | 147.7                  |
| YOLOv5x-seg | 640                   | **50.7**             | **41.4**              | 62:56 (3x)                                    | 1579.2                         | 4.5                            | 88.8               | 265.7                  |

- All checkpoints are trained to 300 epochs with SGD optimizer with `lr0=0.01` and `weight_decay=5e-5` at image size 640 and all default settings.
- **Accuracy** values are for single-model single-scale on COCO dataset.<br>Reproduce by `python segment/val.py --data coco.yaml --weights yolov5s-seg.pdparams`
- **Speed** averaged over 100 inference images using a [Colab Pro](https://colab.research.google.com/signup) A100 High-RAM instance. Values indicate inference speed only (NMS adds about 1ms per image). <br>Reproduce by `python segment/val.py --data coco.yaml --weights yolov5s-seg.pdparams --batch 1`
- **Export** to ONNX at FP32 and TensorRT at FP16 done with `export.py`. <br>Reproduce by `python export.py --weights yolov5s-seg.pdparams --include engine --device 0 --half`

</details>

<details>
  <summary>Segmentation Usage Examples</summary>

### Train

YOLOv5 segmentation training supports auto-download COCO128-seg segmentation dataset with `--data coco128-seg.yaml` argument and manual download of COCO-segments dataset with `bash data/scripts/get_coco.sh --train --val --segments` and then `python train.py --data coco.yaml`.

```bash
# Single-GPU
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pdparams --img 640

# Multi-GPU DDP
python -m paddle.distributed.launch --gpus 0,1,2,3 segment/train.py --weights yolov5s-seg.pdparams --data coco128-seg.yaml --device 0,1,2,3
```

### Val

Validate YOLOv5s-seg mask mAP on COCO dataset:

```bash
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.pdparams --data coco.yaml --img 640  # validate
```

### Predict

Use pretrained YOLOv5m-seg.pdparams to predict bus.jpg:

```bash
python segment/predict.py --weights yolov5m-seg.pdparams --data data/images/bus.jpg
```

| ![zidane](https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg) | ![bus](https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg) |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |

### Export

Export YOLOv5s-seg model to ONNX, TensorRT, etc.

```bash
# export model
python export.py --weights yolov5s-seg.pdparams --include paddleinfer onnx engine openvino paddlelite --img 640 --device 0

# Inference
python detect.py --weights yolov5s.pdparams           # PaddlePaddle
						   yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
						   yolov5s_openvino_model     # OpenVINO
						   yolov5s.engine             # TensorRT
						   yolov5s_paddle_model       # PaddleInference
						   yolov5s.nb                 # PaddleLite
```

</details>

## <div align="center">Classification</div>

YOLOv5 brings support for classification model training, validation and deployment!

<details>
  <summary>Classification Checkpoints</summary>

<br>

We trained YOLOv5-cls classification models on ImageNet for 90 epochs using a 4xA100 instance, and we trained ResNet and EfficientNet models alongside with the same default training settings to compare. We exported all models to ONNX FP32 for CPU speed tests and to TensorRT FP16 for GPU speed tests. We ran all speed tests on Google [Colab Pro](https://colab.research.google.com/signup) for easy reproducibility.

| Model                                                                                              | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Training<br><sup>90 epochs<br>4xA100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TensorRT V100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@224 (B) |
| -------------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | -------------------------------------------- | ------------------------------ | ----------------------------------- | ------------------ | ---------------------- |
| YOLOv5n-cls         | 224                   | 64.6             | 85.4             | 7:59                                         | **3.3**                        | **0.5**                             | **2.5**            | **0.5**                |
| YOLOv5s-cls         | 224                   | 71.5             | 90.2             | 8:09                                         | 6.6                            | 0.6                                 | 5.4                | 1.4                    |
| YOLOv5m-cls         | 224                   | 75.9             | 92.9             | 10:06                                        | 15.5                           | 0.9                                 | 12.9               | 3.9                    |
| YOLOv5l-cls         | 224                   | 78.0             | 94.0             | 11:56                                        | 26.9                           | 1.4                                 | 26.5               | 8.5                    |
| YOLOv5x-cls         | 224                   | **79.0**         | **94.5**         | 15:04                                        | 54.3                           | 1.8                                 | 48.1               | 15.9                   |

<details>
  <summary>Table Notes (click to expand)</summary>

- All checkpoints are trained to 90 epochs with SGD optimizer with `lr0=0.001` and `weight_decay=5e-5` at image size 224 and all default settings.<br>Runs logged to https://wandb.ai/glenn-jocher/YOLOv5-Classifier-v6-2
- **Accuracy** values are for single-model single-scale on [ImageNet-1k](https://www.image-net.org/index.php) dataset.<br>Reproduce by `python classify/val.py --data ../datasets/imagenet --img 224`
- **Speed** averaged over 100 inference images using a Google [Colab Pro](https://colab.research.google.com/signup) V100 High-RAM instance.<br>Reproduce by `python classify/val.py --data ../datasets/imagenet --img 224 --batch 1`
- **Export** to ONNX at FP32 and TensorRT at FP16 done with `export.py`. <br>Reproduce by `python export.py --weights yolov5s-cls.pdparams --include engine onnx --imgsz 224`

</details>
</details>

<details>
  <summary>Classification Usage Examples</summary>

### Train

YOLOv5 classification training supports auto-download of MNIST, Fashion-MNIST, CIFAR10, CIFAR100, Imagenette, Imagewoof, and ImageNet datasets with the `--data` argument. To start training on MNIST for example use `--data mnist`.

```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pdparams --data cifar100 --img 224 --batch 128

# Multi-GPU DDP
python -m paddle.distributed.launch --gpus 0,1,2,3  classify/train.py --model yolov5s-cls.pdparams --data imagenet --img 224 --device 0,1,2,3
```

### Val

Validate YOLOv5m-cls accuracy on ImageNet-1k dataset:

```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pdparams --data ../datasets/imagenet --img 224  # validate
```

### Predict

Use pretrained YOLOv5s-cls.pdparams to predict bus.jpg:

```bash
python classify/predict.py --weights yolov5s-cls.pdparams --data data/images/bus.jpg
```

### Export

Export a group of trained YOLOv5s-cls, ResNet models to ONNX and TensorRT:

```bash
python export.py --weights yolov5s-cls.pdparams resnet50.pdparams --include paddleinfer, onnx, engine, openvino, paddlelite --img 224
```


</details>

