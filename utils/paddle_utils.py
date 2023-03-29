# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license
"""
PaddlePaddle utils
"""

import math
import io
import os
import platform
import subprocess
import time
import pickle
import warnings
import numpy as np
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.framework as framework
from paddle.nn.initializer import Uniform
# from paddle.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe
from utils.profile_utils import paddle_profile # for FLOPs computation

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# Suppress PaddlePaddle warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')
warnings.filterwarnings('ignore', category=UserWarning)


class LabelSmoothingCrossEntropyLoss(nn.Layer):
    # CrossEntropyLoss with label smoothing
    def __init__(self, classes, smoothing=0.0, axis=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.axis = axis

    def forward(self, pred, target):
        pred = F.log_softmax(pred, axis=self.axis)
        with paddle.no_grad():
            true_dist = paddle.ones_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.put_along_axis_(target.unsqueeze(1), self.confidence, 1)
        return paddle.mean(paddle.sum(-true_dist * pred, axis=self.axis))


def clip_grad_norm_(parameters, max_norm, norm_type = 2.0, error_if_nonfinite = False):
    # clip grad by norm
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor([0.])
    if norm_type == 'inf':
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.norm(paddle.stack([paddle.norm(g.detach(), norm_type) for g in grads]), norm_type)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_cliped = paddle.clip(clip_coef, max=1.0)
    for g in grads:
        g = g.detach().multiply(clip_coef_cliped)
    return total_norm


def convert_object_to_tensor(obj):
    # Convert object to tensor
    _pickler = pickle.Pickler
    f = io.BytesIO()
    _pickler(f).dump(obj)
    data = np.frombuffer(f.getvalue(), dtype=np.int32)
    tensor = paddle.to_tensor(data)
    return tensor, tensor.numel()


def convert_tensor_to_object(tensor, len_of_tensor):
    # Convert tensor to object
    _unpickler = pickle.Unpickler
    return _unpickler(io.BytesIO(tensor.numpy()[:len_of_tensor])).load()


def broadcast_object_list(object_list, src, group=None):
    """
    Broadcast picklable objects from the source to all others. Similiar to broadcast(), but python object can be passed in.
    Args:
        object_list (list): The list of objects to send if current rank is the source, or the list of objects to receive otherwise.
        src (int): The source rank in global view.
        group (Group): The group instance return by new_group or None for global default group.
    Returns:
        None.
    Warning:
        This API only supports the dygraph mode.
    Examples:
        .. code-block:: python
            # required: distributed
            import paddle.distributed as dist
            dist.init_parallel_env()
            if dist.get_rank() == 0:
                object_list = [{"foo": [1, 2, 3]}]
            else:
                object_list = [{"bar": [4, 5, 6]}]
            dist.broadcast_object_list(object_list, src=1)
            print(object_list)
            # [{"bar": [4, 5, 6]}] (2 GPUs)
    """
    assert (
        framework.in_dygraph_mode()
    ), "broadcast_object_list doesn't support static graph mode."

    rank = dist.get_rank()
    obj_tensors = []
    obj_nums = len(object_list)

    if rank == src:
        obj_sizes = []
        for obj in object_list:
            obj_tensor, obj_size = convert_object_to_tensor(obj)
            obj_tensors.append(obj_tensor)
            obj_sizes.append(obj_size)
        obj_size_tensor = paddle.concat(obj_sizes)
    else:
        obj_size_tensor = paddle.empty([obj_nums], dtype="int64")
    dist.broadcast(obj_size_tensor, src)

    if rank == src:
        # cast to uint8 to keep the same dtype
        obj_data_tensor = paddle.concat(obj_tensors).cast("int32")
    else:
        data_len = paddle.sum(obj_size_tensor).item()
        obj_data_tensor = paddle.empty([data_len], dtype="int32")
    dist.broadcast(obj_data_tensor, src)

    offset = 0
    for i in range(obj_nums):
        data_len = obj_size_tensor[i]
        object_list[i] = convert_tensor_to_object(
            obj_data_tensor[offset : offset + data_len], data_len
        )
        offset += data_len


def state_dict_float(model_state_dict):
    # Float the model state dict
    float_state_dict = {}
    for key, value in model_state_dict.items():
        if isinstance(value, paddle.Tensor):
            float_state_dict[key] = value.astype(paddle.float32)
    return float_state_dict
    

def model_half(model):
    # Half the model state dict
    model_state_dict = model.state_dict()
    for key, value in model_state_dict.items():
        if 'bn' in key or 'anchors' in key:
            continue
        model_state_dict[key] = value.astype(paddle.float16)
    # Attribute information
    if hasattr(model, 'class_weights'):
        model_state_dict['class_weights'] = model.class_weights
    if hasattr(model, 'hyp'):
        model_state_dict['hyp'] = model.hyp
    if hasattr(model, 'yaml_file'):
        model_state_dict['yaml_file'] = model.yaml_file
    model_state_dict['stride'] = model.stride.cpu().numpy()
    model_state_dict['nc'] = model.nc
    model_state_dict['yaml'] = model.yaml
    model_state_dict['names'] = model.names
    
    return model_state_dict


def smart_inference_mode():
    # Applies paddle.inference_mode() decorator if paddle>=1.9.0 else paddle.no_grad() decorator
    def decorate(fn):
        return (paddle.no_grad)()(fn)

    return decorate


def smartCrossEntropyLoss(classes, label_smoothing=0.0):
    # Returns nn.CrossEntropyLoss with label smoothing enabled for paddle>=2.1.0
    if check_version(paddle.__version__, '2.1.0'):
        return LabelSmoothingCrossEntropyLoss(classes=classes, smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f'WARNING ⚠️ label smoothing {label_smoothing} requires paddle>=1.10.0')
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(paddle.__version__, '2.1.0', pinned=True), \
        'DDP training is not supported due to the low version. ' \
        'Please upgrade or downgrade paddlepaddle to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(paddle.__version__, '2.1.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


@contextmanager
def paddle_distributed_zero_first(rank):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if rank not in [-1, 0]:
        dist.barrier()
    yield
    if rank == 0:
        dist.barrier()


def reshape_classifier_output(model, n=1000):
    # Update a classification model to class count 'n' if required
    from models.common import Classify
    name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module

    if isinstance(m, Classify):  # YOLOv5 Classify() head
        if m.linear.weight.shape[1] != n:
            k = math.sqrt(1.0 / m.linear.weight.shape[0])
            ParamAttr = paddle.ParamAttr(initializer=Uniform(-k, k))
            m.linear = nn.Linear(m.linear.weight.shape[0], n,
                                 weight_attr=ParamAttr,
                                 bias_attr=ParamAttr)
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
        if m.weight.shape[1] != n:
            k = math.sqrt(1.0 / m.weight.shape[0])
            ParamAttr = paddle.ParamAttr(initializer=Uniform(-k, k))
            setattr(model, name, nn.Linear(m.weight.shape[0], n, weight_attr=ParamAttr, bias_attr=ParamAttr))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)  # nn.Linear index
            if m[i].weight.shape[1] != n:
                k = math.sqrt(1.0 / m[i].weight.shape[0])
                ParamAttr = paddle.ParamAttr(initializer=Uniform(-k, k))
                m[i] = nn.Linear(m[i].weight.shape[0], n,
                                 weight_attr=ParamAttr,
                                 bias_attr=ParamAttr)
        elif nn.Conv2D in types:
            i = types.index(nn.Conv2D)  # nn.Conv2D index
            if m[i]._out_channels != n:
                m[i] = nn.Conv2D(m[i]._in_channels, n, m[i]._kernel_size, m[i]._stride, bias_attr=m[i].bias is not None)


def device_count():
    # Returns number of CUDA devices available. Safe version of paddle.device.cuda.device_count(). Supports Linux and Windows
    assert platform.system() in ('Linux', 'Windows'), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} paddle-{paddle.__version__} '
    # device = str(device).strip().lower().replace('place(gpu:', '').replace(')', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force paddle.device.is_compiled_with_cuda() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and paddle.device.is_compiled_with_cuda():  # prefer GPU if available
        
        devices = device.split(',') if device else '0'  # range(paddle.device.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = paddle.device.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = f'gpu:{int(device if device else 0)}' if n in [1, ''] else None
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return paddle.device.set_device(arg) if arg else devices


def time_sync():
    # PaddlePaddle-accurate time
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10):
    """ YOLOv5 speed/memory/FLOPs profiler
    Usage:
        input = paddle.randn(16, 3, 640, 640)
        m1 = lambda x: x * F.sigmoid(x)
        m2 = nn.Silu()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
    """
    results = []
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x.stop_gradient = False
        for m in ops if isinstance(ops, list) else [ops]:
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = paddle_profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = paddle.device.cuda.memory_reserved() / 1E9 if paddle.device.is_compiled_with_cuda() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, paddle.Tensor) else 'list' for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()).item() if isinstance(m, nn.Layer) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            paddle.device.cuda.empty_cache()
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return isinstance(model, paddle.DataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model._layers if is_parallel(model) else model


def initialize_weights(model):
    for m in model.sublayers():
        t = type(m)
        if t is nn.Conv2D:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2D:
            m._epsilon = 1e-3
            m.momentum = 0.97
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.Silu]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2D):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import paddle.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2D):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    LOGGER.info(f'Model pruned to {sparsity(model):.3g} global sparsity')


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2D() and BatchNorm2D() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2D(conv._in_channels,
                          conv._out_channels,
                          kernel_size=conv._kernel_size,
                          stride=conv._stride,
                          padding=conv._padding,
                          dilation=conv._dilation,
                          groups=conv._groups,
                          weight_attr=paddle.ParamAttr(trainable=False),
                          bias_attr=paddle.ParamAttr(trainable=False))

    # Prepare filters
    w_conv = conv.weight.clone().reshape([conv._out_channels, -1])
    w_bn = paddle.diag(bn.weight.divide(paddle.sqrt(bn._epsilon + bn._variance))).astype(paddle.get_default_dtype())
    fusedconv.weight.set_value(paddle.mm(w_bn, w_conv).reshape(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = paddle.zeros([conv.weight.shape[0]], dtype=paddle.get_default_dtype()) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.multiply(bn._mean).divide(paddle.sqrt(bn._variance + bn._epsilon)).astype(paddle.get_default_dtype())
    fusedconv.bias.set_value(paddle.mm(w_bn, b_conv.reshape([-1, 1])).flatten() + b_bn)

    return fusedconv


def model_info(model, verbose=False, check_amp=False, imgsz=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    if check_amp: return
    n_p = int(sum(x.numel() for x in model.parameters()))  # number parameters
    n_g = int(sum(x.numel() for x in model.parameters() if not x.stop_gradient))  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('layer_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, not p.stop_gradient, p.numel(), list(p.shape), p.mean(), p.std()))

    p = model.parameters()[0]
    stride = max(int(max(model.stride.cpu().numpy())), 32) if hasattr(model, 'stride') else 32  # max stride
    im = paddle.empty((1, p.shape[1], stride, stride))  # input image in BCHW format
    flops = paddle_profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
    imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
    fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f'{name} summary: {len(model.sublayers())} layers, {n_p} parameters, {n_g} gradients{fs}')


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k in exclude:
            continue
        else:
            setattr(a, k, v)


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2D()
    for v in model.sublayers():
        for p_name, p in v.named_parameters(include_sublayers=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)
    
    if name == 'Adam':
        optimizer = paddle.optimizer.Adam(parameters=[{'params': g[2]}], learning_rate=lr, beta1=momentum, beta2=0.999)  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = paddle.optimizer.AdamW(parameters=[{'params': g[2]}], learning_rate=lr, beta1=momentum, beta2=0.999, weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = paddle.optimizer.RMSProp(parameters=[{'params': g[2]}], learning_rate=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = paddle.optimizer.Momentum(parameters=[{'params': g[2]}], learning_rate=lr, momentum=momentum, use_nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer._add_param_group({'params': g[0], 'weight_decay': decay, 'grad_clip': None})  # add g0 with weight_decay
    optimizer._add_param_group({'params': g[1], 'weight_decay': 0.0, 'grad_clip': None})  # add g1 (BatchNorm2D weights)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr.base_lr}) with parameter groups "
                f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
    return optimizer


def smart_hub_load(repo='ultralytics/yolov5', model='yolov5s', **kwargs):
    # YOLOv5 paddle.hub.load() wrapper with smart error/issue handling
    if check_version(paddle.__version__, '2.1.0'):
        kwargs['skip_validation'] = True  # validation causes GitHub API rate limit errors
    if check_version(paddle.__version__, '2.1.0'):
        kwargs['trust_repo'] = True  # argument required starting in paddle 0.12
    try:
        return paddle.hub.load(repo, model, **kwargs)
    except Exception:
        return paddle.hub.load(repo, model, force_reload=True, **kwargs)


def smart_resume(ckpt, optimizer, ema=None, weights='yolov5s.pt', epochs=300, resume=True):
    # Resume training from a partially trained checkpoint
    best_fitness = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.set_state_dict(ckpt['optimizer'])  # optimizer
        best_fitness = ckpt['best_fitness']
    if ema and ckpt.get('ema'):
        ema.ema.set_state_dict(state_dict_float(ckpt['ema']))  # EMA
        ema.updates = ckpt['updates']
    if resume:
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.\n' \
                                f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        LOGGER.info(f'Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs')
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt['epoch']  # finetune additional epochs
    return best_fitness, start_epoch, epochs


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
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/PaddlePaddle/PaddleDetection
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model))  # FP32 EMA
        self.ema.eval()
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.stop_gradient = True

    def update(self, model):
        # Update EMA parameters
        state_dict = {}
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.is_floating_point() and k in msd.keys():  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
                state_dict[k] = v
            state_dict[k] = v
        # assert v.dtype == msd[k].dtype == paddle.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'
        self.ema.set_state_dict(state_dict)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
