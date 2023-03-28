# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as Initializer

from utils.downloads import attempt_download


class Sum(nn.Layer):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = -paddle.arange(1.0, n) / 2
            self.w = paddle.create_parameter(shape=self.w.shape, dtype=self.w.dtype,
                                             default_initializer=Initializer.Assign(self.w))  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = F.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Layer):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = paddle.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.LayerList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias_attr=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2D(c2)
        self.act = nn.Silu()

    def forward(self, x):
        return self.act(self.bn(paddle.concat([m(x) for m in self.m], 1)))


class Ensemble(nn.LayerList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = paddle.stack(y).max(0)[0]  # max ensemble
        # y = paddle.stack(y).mean(0)  # mean ensemble
        y = paddle.concat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class Ensemble(nn.LayerList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = paddle.stack(y).max(0)[0]  # max ensemble
        # y = paddle.stack(y).mean(0)  # mean ensemble
        y = paddle.concat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, data=None, inplace=True, fuse=True, fp16=False, verbose=False, check_amp=False):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model, ClassificationModel
    
    paddle.set_default_dtype("float16" if fp16 else "float32")
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        if not isinstance(w, str):
            w = str(w)
        ckpt_weights = paddle.load(attempt_download(w))
        ckpt_weights = ckpt_weights.get('ema') or ckpt_weights['model']  # FP32 model
        ckpt = Model(cfg=ckpt_weights['yaml'], ch=3, nc=ckpt_weights['nc'], verbose=verbose,
                     anchors=ckpt_weights['hyp'].get('anchors') if hasattr(ckpt_weights, 'hyp') else None)

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = paddle.to_tensor([32.])
        if 'yaml_file' not in ckpt_weights.keys():
            yaml = ckpt.yaml
            ckpt = ClassificationModel(model=ckpt, nc=ckpt.nc, cutoff=None or 9) # ClassificationModel
            ckpt.yaml = yaml
        if 'names' in ckpt_weights.keys():
            if isinstance(ckpt_weights['names'], (list, tuple)):
                ckpt.names = dict(enumerate(ckpt_weights['names']))  # convert to dict
            else:
                ckpt.names = ckpt_weights['names']
        if not fp16:
            for key, value in ckpt_weights.items():
                if isinstance(value, paddle.Tensor):
                    ckpt_weights[key] = value.astype(paddle.float32)
        ckpt.set_state_dict(ckpt_weights)
        if fuse and hasattr(ckpt, 'fuse'):
            ckpt.fuse(check_amp).eval()
        else:
            ckpt.eval()
        model.append(ckpt)  # model in eval mode

    # Module compatibility updates
    for m in model.sublayers():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.Silu, Detect, Model):
            m.inplace = inplace  # paddle 2.1.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [paddle.zeros([1])] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # paddle 2.1.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[paddle.argmax(paddle.to_tensor([m.stride.max() for m in model], dtype=paddle.int32))].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model
