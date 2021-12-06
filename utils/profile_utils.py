# YOLOv5 reproduction 🚀 by GuoQuanhao

import paddle
import paddle.nn as nn
from paddle.nn.layer.conv import _ConvNd
import logging


def prRed(skk):
    print("\033[91m{}\033[00m".format(skk))

def counter_linear(in_feature, num_elements):
    return paddle.to_tensor([int(in_feature * num_elements)], dtype=paddle.float64)

def counter_upsample(mode: str, output_size):
    total_ops = output_size
    if mode == "linear":
        total_ops *= 5
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "bicubic":
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops *= ops_solve_A + ops_solve_p
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return paddle.to_tensor([int(total_ops)], dtype=paddle.float64)

def counter_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return paddle.to_tensor([int(total_ops)], dtype=paddle.float64)

def counter_relu(input_size: paddle.Tensor):
    return paddle.to_tensor([int(input_size)], dtype=paddle.float64)

def counter_avgpool(input_size):
    return paddle.to_tensor([int(input_size)], dtype=paddle.float64)

def counter_adap_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return paddle.to_tensor([int(kernel_op * output_size)], dtype=paddle.float64)

def counter_norm(input_size):
    """input is a number not a array or tensor"""
    return paddle.to_tensor([2 * int(input_size)], dtype=paddle.float64)

def counter_parameters(para_list):
    total_params = 0
    for p in para_list:
        total_params += paddle.to_tensor([int(p.numel())], dtype=paddle.float64)
    return total_params

def counter_conv(bias, kernel_size, output_size, in_channel, group):
    """inputs are all numbers!"""
    return paddle.to_tensor([int(output_size * (in_channel / group * kernel_size + bias))])

def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += paddle.to_tensor([p.numel().item()], dtype=paddle.float64)
    m.total_params[0] = counter_parameters(m.parameters())

def count_adap_avgpool(m, x, y):
    kernel = paddle.to_tensor(
        [*(x[0].shape[2:])], dtype=paddle.float64) // paddle.to_tensor([*(y.shape[2:])], dtype=paddle.float64)
    total_add = paddle.prod(kernel)
    num_elements = y.numel().item()
    m.total_ops += counter_adap_avg(total_add, num_elements)

def count_avgpool(m, x, y):
    # total_add = paddle.prod(paddle.to_tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel().item()
    m.total_ops += counter_avgpool(num_elements)

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel().item()

    m.total_ops += counter_relu(nelements)

def count_softmax(m, x, y):
    x = x[0]
    nfeatures = x.shape[m.dim]
    batch_size = x.numel().item() //nfeatures

    m.total_ops += counter_softmax(batch_size, nfeatures)

def counter_zero_ops():
    return paddle.to_tensor([int(0)], dtype=paddle.float64)

def count_convNd(m: _ConvNd, x: (paddle.Tensor,), y: paddle.Tensor):
    x = x[0]

    kernel_ops = paddle.zeros(m.weight.shape[2:]).numel().item()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    m.total_ops += counter_conv(bias_ops, paddle.zeros(m.weight.shape
                                [2:]).numel().item(), len(y.flatten()), m._in_channels, m._groups)

def count_prelu(m, x, y):
    x = x[0]

    nelements = x.numel().item()
    if not m.training:
        m.total_ops += counter_relu(nelements)

def count_in(m, x, y):
    x = x[0]
    if not m.training:
        m.total_ops += counter_norm(x.numel())

def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logging.warning(
            "mode %s is not implemented yet, take it a zero op" % m.mode)
        return counter_zero_ops()

    if m.mode == "nearest":
        return counter_zero_ops()

    x = x[0]
    m.total_ops += counter_upsample(m.mode, y.numel())

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel().item()

    m.total_ops += counter_linear(total_mul, num_elements)

def count_ln(m, x, y):
    x = x[0]
    if not m.training:
        m.total_ops += counter_norm(x.numel())

def count_bn(m, x, y):
    x = x[0]
    if not m.training:
        m.total_ops += counter_norm(x.numel())

def zero_ops(m, x, y):
    m.total_ops += counter_zero_ops()


register_hooks = {
    nn.Pad1D: zero_ops, # padding does not involve any multiplication.
    nn.Pad2D: zero_ops,
    nn.Pad3D: zero_ops,

    nn.Conv1D: count_convNd,
    nn.Conv2D: count_convNd,
    nn.Conv3D: count_convNd,
    nn.Conv1DTranspose: count_convNd,
    nn.Conv2DTranspose: count_convNd,
    nn.Conv3DTranspose: count_convNd,

    nn.SyncBatchNorm: count_bn,

    nn.BatchNorm1D: count_bn,
    nn.BatchNorm2D: count_bn,
    nn.BatchNorm3D: count_bn,
    nn.LayerNorm: count_ln,
    nn.InstanceNorm1D: count_in,
    nn.InstanceNorm2D: count_in,
    nn.InstanceNorm3D: count_in,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1D: zero_ops,
    nn.MaxPool2D: zero_ops,
    nn.MaxPool3D: zero_ops,
    nn.AdaptiveMaxPool1D: zero_ops,
    nn.AdaptiveMaxPool2D: zero_ops,
    nn.AdaptiveMaxPool3D: zero_ops,

    nn.AvgPool1D: count_avgpool,
    nn.AvgPool2D: count_avgpool,
    nn.AvgPool3D: count_avgpool,
    nn.AdaptiveAvgPool1D: count_adap_avgpool,
    nn.AdaptiveAvgPool2D: count_adap_avgpool,
    nn.AdaptiveAvgPool3D: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2D: count_upsample,
    nn.UpsamplingNearest2D: count_upsample,

    # nn.GRUCell: count_gru_cell,
    # nn.LSTMCell: count_lstm_cell,
    # nn.RNN: count_rnn,
    # nn.GRU: count_gru,
    # nn.LSTM: count_lstm,
    nn.Sequential: zero_ops,

}


def paddle_profile(model: nn.Layer, inputs, custom_ops=None, verbose=True, ret_layer_info=False, report_missing=False):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Layer):
        m.register_buffer('total_ops', paddle.zeros([1], dtype=paddle.float64))
        m.register_buffer('total_params', paddle.zeros([1], dtype=paddle.float64))

        m_type = type(m)

        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and report_missing:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handler_collection[m] = (m.register_forward_post_hook(fn), m.register_forward_post_hook(count_parameters))
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with paddle.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Layer, prefix="\t") -> (int, int):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            next_dict = {}
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.LayerList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params
        # print(prefix, module._get_name(), (total_ops, total_params))
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model)

    # reset model to original status
    if prev_training_status:
        model.train()
    else:
        model.eval()
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params
