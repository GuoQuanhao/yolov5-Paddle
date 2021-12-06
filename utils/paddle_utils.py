# YOLOv5 🚀 reproduction by Quanhao Guo
"""
Paddle utils
"""

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import GPUtil

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from utils.profile_utils import paddle_profile


@contextmanager
def paddle_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(group=local_rank)
    yield
    if local_rank == 0:
        dist.barrier(group=0)


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} paddle {paddle.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force paddle.device.is_compiled_with_cuda() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert paddle.device.is_compiled_with_cuda(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and paddle.device.is_compiled_with_cuda()
    if cuda:
        devices = device.split(',') if device else '0'  # range(len(GPUtil.getGPUs()))  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        all_GPUs = GPUtil.getGPUs()
        select_GPUs = []
        for id in device:
            select_GPUs.append(all_GPUs[eval(id)])
        for i, d in enumerate(select_GPUs):
            s += f"{'' if i == 0 else space}CUDA:{d.id} ({d.name}, {d.memoryTotal}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    print(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return paddle.CUDAPlace(0) if cuda else paddle.CPUPlace()


def time_sync():
    return time.time()


def profile(input, ops, n=10):
    # YOLOv5 speed/memory/FLOPs profiler
    #
    # Usage:
    #     input = paddle.randn([16, 3, 640, 640])
    #     m1 = lambda x: x * paddle.nn.functional.sigmoid(x)
    #     m2 = nn.Silu()
    #     profile(input, [m1, m2], n=100)  # profile over 100 iterations

    results = []
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x.stop_gradient = False
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.half() if hasattr(m, 'half') and isinstance(x, paddle.Tensor) and x.dtype is paddle.float16 else m
            tf, tb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
            try:
                flops = paddle_profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception as e:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                GPU = GPUtil.getGPUs()[0]
                mem = GPU.memoryUsed / 1024 if paddle.device.is_compiled_with_cuda() else 0  # (GB)
                s_in = tuple(x.shape) if isinstance(x, paddle.Tensor) else 'list'
                s_out = tuple(y.shape) if isinstance(y, paddle.Tensor) else 'list'
                p = int(sum(list(x.numel() for x in m.parameters()))) if isinstance(m, nn.Layer) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) is paddle.fluid.dygraph.parallel.DataParallel


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model._layer if is_parallel(model) else model


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.sublayers():
        t = type(m)
        if t is nn.Conv2D:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2D:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.Silu]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2D):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.layer_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import paddle.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_sublayers():
        if isinstance(m, nn.Conv2D):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2D(conv._in_channels,
                          conv._out_channels,
                          kernel_size=conv._kernel_size,
                          stride=conv._stride,
                          padding=conv._padding,
                          groups=conv._groups,
                          bias_attr=None)

    for param in fusedconv.parameters():
        param.stop_gradient = True

    # prepare filters
    w_conv = conv.weight.clone().reshape([conv._out_channels, -1])
    w_bn = paddle.diag(bn.weight.divide(paddle.sqrt(bn.eps + bn._variance)))
    fusedconv.weight.set_value(paddle.mm(w_bn, w_conv).reshape(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = paddle.zeros([conv.weight.shape[0]]) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.multiply(bn._mean).divide(paddle.sqrt(bn._variance + bn.eps))
    fusedconv.bias.set_value(paddle.mm(w_bn, b_conv.reshape([-1, 1])).flatten() + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = int(sum(x.numel() for x in model.parameters()))  # number parameters
    n_g = int(sum(x.numel() for x in model.parameters() if not x.stop_gradient))  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('layer_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.stop_gradient, p.numel(), list(p.shape), p.mean(), p.std()))
    
    stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
    img = paddle.zeros((1, model.yaml.get('ch', 3), stride, stride))  # input
    flops = paddle_profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
    fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    print(f"Model Summary: {len(list(model.sublayers()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = paddle.vision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = paddle.create_parameter(shape=[n], dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))
    model.fc.weight = paddle.create_parameter(shape=[n, filters], dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
    a.class_weights = b.class_weights


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model._layers if is_parallel(model) else model)  # FP32 EMA
        self.ema.eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.stop_gradient = True

    def update(self, model):
        # Update EMA parameters
        state_dict = {}
        with paddle.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model._layers.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype in [paddle.float16, paddle.float32]:
                    v *= d
                    v += (1. - d) * msd[k].detach()
                    state_dict[k] = v
                state_dict[k] = v
            self.ema.set_state_dict(state_dict)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
