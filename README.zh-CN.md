<div align="center">

  <a href="https://github.com/GuoQuanhao/YOLOv5-Paddle" target="_blank">
    <img width="1024", src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/imgs/logo.png"></a>

<br>
  <a href="https://aistudio.baidu.com/aistudio/projectdetail/2580805?contributionType=1"><img width="100" src="https://raw.githubusercontent.com/GuoQuanhao/yolov5-Paddle/main/data/assets/AIStudio.png" alt="Run on AIstudio"></a>
<br>

[English](README.md) | [简体中文](README.zh-CN.md)

这篇由<a href="https://github.com/GuoQuanhao">郭权浩</a>创建的<a href="https://github.com/GuoQuanhao/yolov5-Paddle">YOLOv5-Paddle 🚀</a>笔记本提供了简单的训练、验证、预测和模型导出示例。YOLOv5-Paddle先在支持多种格式的单精度和半精度模型导出。 在代码运行上遇到问题请联系<a href="https://github.com/GuoQuanhao">我</a>以提供便捷的支持。
<br>
<br>&#9745; [PaddleLite](https://www.paddlepaddle.org.cn/lite/v2.12/api_reference/python_api/opt.html) &#9745; [PaddleInference](https://www.paddlepaddle.org.cn/inference/v2.4/guides/export_model/paddle_model_export.html) &#9745; [ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) &#9745; [OpenVIVO](https://github.com/openvinotoolkit/openvino) &#9745; [TensorRT](https://github.com/PaddlePaddle/Paddle2ONNX)<br>
</div>

## <div align="center">目标检测</div>

<details open>
<summary>安装</summary>

克隆仓库并安装 [requirements.txt](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/requirements.txt)，并且
[**Python>=3.7.0**](https://www.python.org/)，[**PaddlePaddle>=2.4.0**](https://www.paddlepaddle.org.cn/).

```bash
git clone https://github.com/GuoQuanhao/yolov5-Paddle  # clone
cd yolov5-Paddle
pip install -r requirements.txt  # install
```


<details>
<summary>推理</summary>

YOLOv5 PaddlePaddle inference. Models download automatically from the latest

```python
# Model
python hubconf.py  # or yolov5n - yolov5x6, custom
```

`detect.py` 能够利用`--source`指定各种媒体资源，并自动从百度云智能云服务器下载PaddlePaddle模型，并将检测结果保存在`runs/detect`。

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
<summary>训练</summary>

下方命令能够重现YOLOv5的 [COCO](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/scripts/get_coco.sh) 结果. [模型](https://github.com/GuoQuanhao/yolov5-Paddle/tree/main/models) 和 [数据集](https://github.com/GuoQuanhao/yolov5-Paddle/tree/main/data) 能够重最新的YOLOv5 [release](https://github.com/GuoQuanhao/yolov5-Paddle/releases/tag/v2.0)中自动下载。 下面展示了在V100-16GB上的Batch sizes。

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
<summary>评估</summary>

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
  <summary>YOLOv5 与 YOLOv5-P5 640</summary>
<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>

<details>
  <summary>图片中的注意事项</summary>

- **COCO AP val** 表示在 5000 张图像的 [COCO val2017](http://cocodataset.org) 数据集上测量的 mAP@0.5:0.95 指标，推理大小从 256 到 1536。
- **GPU Speed** 使用批量大小为 32 的 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 实例测量 [COCO val2017](http://cocodataset.org) 数据集上每张图像的平均推理时间。
- **Reproduce** 通过 `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pdparams yolov5s6.pdparams yolov5m6.pdparams yolov5l6.pdparams yolov5x6.pdparams`

</details>

<details>
 <summary>目标检测模型权重</summary>
PaddlePaddle实现了对精度、参数和flops的验证，未利用PaddlePaddle验证推理速度。

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
  <summary>表格注意事项</summary>

- 所有模型都采用默认设置训练300轮. Nano和Small模型采用[hyp.scratch-low.yaml](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/hyps/hyp.scratch-low.yaml) hyps, 其余模型采用 [hyp.scratch-high.yaml](https://github.com/GuoQuanhao/yolov5-Paddle/blob/main/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** 值适用于 [COCO val2017](http://cocodataset.org)数据集上的单模型单尺度。<br>复现示例：`python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **Speed** 使用 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) 实例对 COCO val 图像进行平均。 NMS 时间 (~1 ms/img) 不包括在内。<br>复现示例：`python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [Test Time Augmentation](https://github.com/ultralytics/yolov5/issues/303) 包括反射和尺度增强。<br>复现示例：`python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

<details>
<summary>模型导出</summary>

```bash
python export.py --weights yolov5n.pdparams --include paddleinfer onnx engine openvino paddlelite
						   yolov5s.pdparams
						   yolov5m.pdparams
						   yolov5l.pdparams
						   yolov5x.pdparams
```
你可以使用 `--dynamic` or `--half` 来导出动态维度或半精度模型。
</details>

## <div align="center">Segmentation</div>

<details>
  <summary>图像分割模型权重</summary>

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

- 所有权重采用 `lr0=0.01` 的SGD优化器训练300轮，其中`weight_decay=5e-5`，图像尺寸为640×640。
- **Accuracy** 评估于COCO数据集上的单模型，单尺度。<br>复现示例：`python segment/val.py --data coco.yaml --weights yolov5s-seg.pdparams`
- **Speed** 使用 [Colab Pro](https://colab.research.google.com/signup) A100 High-RAM 实例对超过 100 张推理图像进行平均。 值仅表示推理速度（NMS 每张图像增加约 1 毫秒）。 <br>复现示例：`python segment/val.py --data coco.yaml --weights yolov5s-seg.pdparams --batch 1`
- **Export** 为FP32的ONNX模型和FP16的TensorRT模型。 <br>复现示例：`python export.py --weights yolov5s-seg.pdparams --include engine --device 0 --half`

</details>

<details>
  <summary>图像分割使用示例</summary>

### Train

YOLOv5 分割训练支持使用 `--data coco128-seg.yaml` 参数自动下载 COCO128-seg 分割数据集和使用 `bash data/scripts/get_coco.sh --train --val -- segments`手动下载 COCO-segments 数据集。`python train.py --data coco.yaml`。
```bash
# Single-GPU
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pdparams --img 640

# Multi-GPU DDP
python -m paddle.distributed.launch --gpus 0,1,2,3 segment/train.py --weights yolov5s-seg.pdparams --data coco128-seg.yaml --device 0,1,2,3
```

### Val

在 COCO 数据集上验证 YOLOv5s-seg 的 mask mAP：

```bash
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.pdparams --data coco.yaml --img 640  # validate
```

### Predict

使用预训练的 YOLOv5m-seg.pdparams 预测 bus.jpg：

```bash
python segment/predict.py --weights yolov5m-seg.pdparams --data data/images/bus.jpg
```

| ![zidane](https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg) | ![bus](https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg) |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |

### Export

导出ONNX, TensorRT等分割模型。

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

## <div align="center">分类模型</div>

YOLOv5 带来了对分类模型训练、验证和部署的支持！

<details>
  <summary>分类模型权重</summary>

<br>

| Model                                                                                              | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Training<br><sup>90 epochs<br>4xA100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TensorRT V100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@224 (B) |
| -------------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | -------------------------------------------- | ------------------------------ | ----------------------------------- | ------------------ | ---------------------- |
| YOLOv5n-cls         | 224                   | 64.6             | 85.4             | 7:59                                         | **3.3**                        | **0.5**                             | **2.5**            | **0.5**                |
| YOLOv5s-cls         | 224                   | 71.5             | 90.2             | 8:09                                         | 6.6                            | 0.6                                 | 5.4                | 1.4                    |
| YOLOv5m-cls         | 224                   | 75.9             | 92.9             | 10:06                                        | 15.5                           | 0.9                                 | 12.9               | 3.9                    |
| YOLOv5l-cls         | 224                   | 78.0             | 94.0             | 11:56                                        | 26.9                           | 1.4                                 | 26.5               | 8.5                    |
| YOLOv5x-cls         | 224                   | **79.0**         | **94.5**         | 15:04                                        | 54.3                           | 1.8                                 | 48.1               | 15.9                   |

<details>
  <summary>表格注意事项</summary>

- 所有检查点都使用 SGD 优化器训练 90 轮，其中`lr0=0.001`，`weight_decay=5e-5`，图像大小为 224。
- **Accuracy** values are for single-model single-scale on [ImageNet-1k](https://www.image-net.org/index.php) dataset.<br>复现示例：`python classify/val.py --data ../datasets/imagenet --img 224`
- **Speed** 使用 Google [Colab Pro](https://colab.research.google.com/signup) V100 High-RAM 实例计算 100 多张推理图像的平均值。<br>复现示例：`python classify/val.py --data ../datasets/imagenet --img 224 --batch 1`
- **Export** 为FP32的ONNX模型和FP16的TensorRT模型。 <br>复现示例：`python export.py --weights yolov5s-cls.pdparams --include engine onnx --imgsz 224`

</details>
</details>

<details>
  <summary>图像分类使用示例</summary>

### Train

YOLOv5 分类训练支持使用`--data`参数自动下载 MNIST、Fashion-MNIST、CIFAR10、CIFAR100、Imagenette、Imagewoof 和 ImageNet 数据集。 例如，要在 MNIST 上开始训练，请使用 `--data mnist`。
```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pdparams --data cifar100 --img 224 --batch 128

# Multi-GPU DDP
python -m paddle.distributed.launch --gpus 0,1,2,3  classify/train.py --model yolov5s-cls.pdparams --data imagenet --img 224 --device 0,1,2,3
```

### Val

在 ImageNet-1k 数据集上验证 YOLOv5m-cls 的准确性：

```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pdparams --data ../datasets/imagenet --img 224  # validate
```

### Predict

使用预训练的 YOLOv5s-cls.pdparams 来预测 bus.jpg：

```bash
python classify/predict.py --weights yolov5s-cls.pdparams --data data/images/bus.jpg
```

### Export

将一组训练好的 YOLOv5s-cls、ResNet 模型导出到 ONNX 和 TensorRT：

```bash
python export.py --weights yolov5s-cls.pdparams resnet50.pdparams --include paddleinfer, onnx, engine, openvino, paddlelite --img 224
```


</details>
