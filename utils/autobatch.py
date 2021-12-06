# YOLOv5 reproduction 🚀 by GuoQuanhao
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import paddle
from paddle import amp
import GPUtil

from utils.general import colorstr
from utils.paddle_utils import profile


def check_train_batch_size(model, imgsz=640):
    # Check YOLOv5 training batch size
    with amp.auto_cast():
        copy_model = deepcopy(model)
        copy_model.train()
        return autobatch(copy_model, imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    # Automatically estimate best batch size to use `fraction` of available CUDA memory
    # Usage:
    #     import paddle
    #     from utils.autobatch import autobatch
    #     model = paddle.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    prefix = colorstr('autobatch: ')
    print(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = GPUtil.getGPUs()[0]
    if not device:
        print(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size

    d = device.name.upper()  # 'CUDA:0'
    t = device.memoryTotal / 1024  # (GB)
    r = device.memoryUtil / 1024  # (GB)
    a = device.memoryUsed / 1024  # (GB)
    f = device.memoryFree / 1024  # free inside reserved
    print(f'{prefix}{d} {t:.3g}G total, {r:.3g}G reserved, {a:.3g}G allocated, {f:.3g}G free')

    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [paddle.zeros([b, 3, imgsz, imgsz]) for b in batch_sizes]
        y = profile(img, model, n=3)
    except Exception as e:
        print(f'{prefix}{e}')

    y = [x[2] for x in y if x]  # memory [2]
    batch_sizes = batch_sizes[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    print(f'{prefix}Using colorstr(batch-size {b}) for {d} {t * fraction:.3g}G/{t:.3g}G ({fraction * 100:.0f}%)')
    return b
