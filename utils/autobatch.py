# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import paddle

from utils.general import LOGGER, colorstr
from utils.paddle_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    # Check YOLOv5 training batch size
    with paddle.amp.auto_cast(enable=amp):
        return autobatch(deepcopy(model), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Automatically estimate best YOLOv5 batch size to use `fraction` of available CUDA memory
    # Usage:
    #     import paddle
    #     from utils.autobatch import autobatch
    #     model = paddle.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))
    
    model.train()
    # Check device
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = paddle.device.get_device()  # get model device
    if 'gpu' not in device:
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = 'CUDA:' + device.split(':')[-1]  # 'CUDA:0'
    properties = paddle.device.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = paddle.device.cuda.memory_reserved(device) / gb  # GiB reserved
    a = paddle.device.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [paddle.empty([b, 3, imgsz, imgsz]) for b in batch_sizes]
        results = profile(img, model, n=3)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f'{prefix}WARNING ⚠️ CUDA anomaly detected, recommend restart environment and retry command.')

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    LOGGER.info(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b
