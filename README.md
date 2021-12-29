# YOLOv5-Paddle
YOLOv5🚀 reproduction by Guo Quanhao using PaddlePaddle


[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=PaddlePaddle&repo=Paddle)](https://github.com/PaddlePaddle/Paddle)

- <font face="微软雅黑" color=red size=3>支持</font><font face="Times New Roman" color=red size=3>AutoBatch</font>
- <font face="微软雅黑" color=red size=3>支持</font><font face="Times New Roman" color=red size=3>AutoAnchor</font>
- <font face="微软雅黑" color=red size=3>支持</font><font face="Times New Roman" color=red size=3>GPU Memory</font>

## 快速开始

**使用AIStudio高性能环境快速构建YOLOv5训练(PaddlePaddle2.2.0-gpu version)**

需要安装额外模块
```
pip install gputil==1.4.0
pip install pycocotools
```

##### COCO数据集

数据集已挂载至aistudio项目中，如果需要本地训练可以从这里下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/105347)，和[标签](https://aistudio.baidu.com/aistudio/datasetdetail/103218)文件

```
Data
|-- coco
|   |-- annotions
|   |-- images
|      |-- train2017
|      |-- val2017
|      |-- test2017
|   |-- labels
|      |-- train2017
|      |-- val2017
|      |-- train2017.cache(初始解压可删除，训练时会自动生成)
|      |-- val2017.cache(初始解压可删除，训练时会自动生成)
|   |-- test-dev2017.txt
|   |-- val2017.txt
|   |-- train2017.txt
`   `-- validation
```
修改`data/coco.yaml`配置自己的coco路径，你可能需要修改`path`变量
```shell
path: /home/aistudio/Data/coco  # dataset root dir
```

#### 训练
- **考虑到AIStudio对于github的访问速度，预先提供了`Arial.ttf`**

- **AIStudio后端不支持绘图，部分可视乎在AIStudio仓库被注释**

**training scratch for coco**

```shell
mkdir /home/aistudio/.config/QuanhaoGuo/
cp /home/aistudio/Arial.ttf /home/aistudio/.config/QuanhaoGuo/
cd YOLOv5-Paddle
python train.py --img 896 --batch 8 --epochs 300 --data ./data/coco.yaml --cfg yolov5s.yaml --weights ''
```

#### 验证
```shell
python val.py --img 640  --data ./data/coco.yaml --weights ./weights/yolov5s.pdparams --cfg yolov5s.yaml
```
通过`--task [val/test]`控制验证集和测试集

**所有提供的模型验证精度如下，本仓库的所有资源文件包括预训练模型均可在**<font face="微软雅黑" color=red size=5>[百度云盘下载](https://pan.baidu.com/s/1sMY1ABEM7SVqk02jETNodg)</font>code:dng9

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |params<br><sup>(M) |FLOPs<br><sup>@640 (B)|mAP<sup>test<br>0.5:0.95 |mAP<sup>test<br>0.5 
|---  |---  |---    |---    |---    |---|---    |---
|YOLOv5n|640  |28.4   |46.5   |**1.9**|**4.5**   |28.1  |46.2
|YOLOv5s|640  |37.2   |56.4   |7.2    |16.5   |37.1  |56.1
|YOLOv5m|640  |45.1   |64.2   |21.2   |49.0   |45.4  |64.3
|YOLOv5l|640  |48.6   |67.4   |46.5   |109.1   |48.9  |67.5
|YOLOv5x|640  |50.6   |69.1   |86.7   |205.7   |50.7  |69.0
|      |     |       |       |       |   |   |
|YOLOv5n6|1280 |34.0   |51.1   |3.2    |4.6   |34.3  |51.7
|YOLOv5s6|1280 |44.5   |63.4   |16.8   |12.6   |44.3  |63.0
|YOLOv5m6|1280 |50.9   |69.4   |35.7   |50.0   |51.1  |69.5
|YOLOv5l6|1280 |53.5   |71.8   |76.8   |111.4   |53.7  |71.8
|YOLOv5x6<br>+ [TTA][TTA]|1280<br>1536 |54.6<br>**55.2** |72.6<br>**73.0** |140.7<br>- |209.8<br>-   |55.0<br>**55.8**  |73.0<br>**73.5**

**使用本地环境快速构建YOLOv5训练(PaddlePaddle2.2.0-gpu version)**
```shell
git clone https://github.com/GuoQuanhao/YOLOv5-Paddle
```
然后按照**使用AIStudio高性能环境快速构建YOLOv5训练**执行

  
## 训练Custom Data
这里以一个类别的光栅数据集为例，数据集已上传至[AIStudio](https://aistudio.baidu.com/aistudio/datasetdetail/114547)

其组织结构如下：
```
Data
|-- guangshan
|   |-- images
|      |-- train
|      |-- val
|   |-- labels
|      |-- train
|      |-- val
```
另外你需要构建`data/guangshan.yaml`，相关文件已放入相关目录，主要用于指定数据集读取路径和模型配置。
```
# YOLOv5 reproduction 🚀 by GuoQuanhao

train: /home/aistudio/guangshan/images/train  # 118287 images
val: /home/aistudio/guangshan/images/val  # 5000 images
# number of classes
nc: 1
# class names
names: ['spectrum']
```

#### 训练
```shell
python train.py --img 640 --batch 16 --epochs 100 --data ./data/guangshan.yaml --cfg yolov5s.yaml --weights ./weights/yolov5s.pdparams
```
```
Starting training for 100 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      0/99     4.19G    0.1039   0.04733         0        29       640: 100%|████████████████████████████████████████████████████████████████████| 9/9 [01:43<00:00, 11.50s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.64s/it]
                 all         16         29      0.266      0.379      0.226     0.0468

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      1/99     4.41G   0.08177    0.0289         0        37       640: 100%|████████████████████████████████████████████████████████████████████| 9/9 [01:40<00:00, 11.20s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.49s/it]
                 all         16         29      0.462      0.445      0.398      0.109
......
```
完整的训练日志存在`data/training.txt`

**利用VisualDL可视化训练过程**
  
```
visualdl --logdir ./runs/train/exp
```

![](https://ai-studio-static-online.cdn.bcebos.com/674883338e9a42239cdfd2b3e65e9295acc516a368d344f78b0e55fa5d31f006)
  

#### 验证
```shell
python val.py --img 640  --data ./data/guangshan.yaml --cfg yolov5s.yaml --weights ./runs/train/exp/weights/best.pdparams
```

#### 推理
```shell
python detect.py --weights ./runs/train/exp/weights/best.pdparams --cfg yolov5s.yaml --data ./data/guangshan.yaml --source ./data/images/guangshan.jpg
```
 <center><img src="https://ai-studio-static-online.cdn.bcebos.com/24fa2600cbf64c5584ffa6993efe5bf57c01a293d8a84b76a7a9eec0620c3fde" width="400"/><img src="https://ai-studio-static-online.cdn.bcebos.com/662186a1ae414e69aa4d4a5055881cb4fcdcba90318e497d8cf674547c16b956" width="400"/></center>

## TODO 

- Multi-GPU Training ☘️
- PaddleLite inference 🌟
- Model to ONNX ⭐

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| github        | [Deep Hao的github](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
