# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license
"""
Export a YOLOv5 PaddlePaddle model to other formats

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PaddlePaddle                | -                             | yolov5s.pdparams
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
PaddleInference             | `paddle`                      | yolov5s_paddle_model/
PaddleLite                  | `nb`                          | yolov5s.nb

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pdparams --include paddleinfer, onnx, engine, openvino, paddlelite ...

Inference:
    $ python detect.py --weights yolov5s.pdparams           # PaddlePaddle
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s_paddle_model       # PaddleInference
                                 yolov5s.nb                 # PaddleLite
                                 

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import shutil
import warnings
from pathlib import Path
from copy import deepcopy

import pandas as pd
import paddle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_save)
from utils.paddle_utils import select_device, smart_inference_mode

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = '0'

def export_formats():
    # YOLOv5 export formats
    x = [
        ['PaddlePaddle', '-', '.pdparams', True, True],                        # Support FP16 half-precision
        ['PaddleInference', 'paddleinfer', '_paddle_model', True, True], # Not support FP16 half-precision
        ['ONNX', 'onnx', '.onnx', True, True],                           # Support FP16 half-precision
        ['TensorRT', 'engine', '.engine', False, True],                  # Support FP16 half-precision
        ['OpenVINO', 'openvino', '_openvino_model', True, False],        # Support FP16 half-precision
        ['PaddleLite', 'paddlelite', '.nb', True, False]]                # Support FP16 half-precision
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_paddleinfer(model, im, file, prefix=colorstr('PaddleInference:')):
    # YOLOv5 PaddleInference export
    LOGGER.info(f'\n{prefix} starting export with PaddlePaddle {paddle.__version__}...')
    f = str(file).replace('.pdparams', f'_paddle_model{os.sep}')
    InputSpec = [None] + im.shape[1:]
    model = paddle.jit.to_static(deepcopy(model), input_spec=[paddle.static.InputSpec(shape=InputSpec)])
    path = f + file.name.split('.')[0]
    paddle.jit.save(model, path)

    return f, path


@try_export
def export_paddlelite(model, im, file, place, mtype, prefix=colorstr('PaddleLite:')):
    version = platform.python_version()
    try:
        import paddlelite.lite as lite
    except Exception:
        if version[:3] != '3.8':
            check_requirements('paddlelite')
        else:
            LOGGER.info(f'Python {version} doesn\'t have PaddleLite compiled whl package, please refer xxx')
        import paddlelite.lite as lite

    f, path = export_paddleinfer(model, im, file)
    file = str(file.with_suffix(''))
    opt=lite.Opt()
    opt.set_model_file(path + '.pdmodel')
    opt.set_param_file(path + '.pdiparams')
    opt.set_valid_places(place)
    opt.set_model_type(mtype)
    opt.set_optimize_out(file)
    opt.run() # unsupported ops 'silu', compile developed version
    f = Path(file + '.nb')

    return f, None


@try_export
def export_onnx(model, im, file, opset, half, dynamic, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    # https://github.com/PaddlePaddle/Paddle2ONNX
    f, path = export_paddleinfer(model, im, file)
    name = Path(path).name
    check_requirements('paddle2onnx')
    args = [
        'paddle2onnx',
        '--model_dir',
        f, # paddleinfer model directory
        '--model_filename',
        name + '.pdmodel',
        '--params_filename',
        name + '.pdiparams',
        '--save_file',
        str(file.with_suffix('.onnx')),
        '--enable_onnx_checker',
        'True',
        '--export_fp16_model',
        'True' if half else 'False',
        '--opset_version',
        str(opset)
        ]
    subprocess.run(args, check=True, env=os.environ)

    if not dynamic:
        args = [
            'python',
            '-m',
            'paddle2onnx.optimize',
            '--input_model',
            str(file.with_suffix('.onnx')),
            '--output_model',
            str(file.with_suffix('.onnx')),
            '--input_shape_dict',
            f"{{'x':{im.shape}}}"
            ]
        subprocess.run(args, check=True, env=os.environ)
    f = str(file.with_suffix('.onnx'))
    LOGGER.info('For more information, please refer https://github.com/PaddlePaddle/Paddle2ONNX')
    return f, None


@try_export
def export_openvino(model, im, file, metadata, half, prefix=colorstr('OpenVINO:')):
    # YOLOv5 OpenVINO export
    check_requirements('openvino-dev')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.inference_engine as ie # windows ImportError: DLL load failed while importing ie_api
                                           # https://github.com/openvinotoolkit/openvino/issues/7502
    export_onnx(model, im, file, 12, half, False)  # opset 12
    LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
    f = str(file).replace('.pdparams', f'_openvino_model{os.sep}')

    args = [
        'mo',
        '--input_model',
        str(file.with_suffix('.onnx')),
        '--output_dir',
        f,
        '--data_type',
        ('FP16' if half else 'FP32'),]
    subprocess.run(args, check=True, env=os.environ)  # export
    yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_engine(model, im, file, half, dynamic, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    assert im.place.is_gpu_place(), 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt

    if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, half, dynamic)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, half, dynamic)  # opset 12
    onnx = file.with_suffix('.onnx')
    onnx = Path('new_model.onnx')
    LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument')
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return f, None


@smart_inference_mode()
def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pdparams',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        mtype='naive_buffer',
        place='x86', # arm, x86, opencl, npu
        include=('paddleinfer', 'onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        dynamic=False,  # ONNX/TensorRT: dynamic axes
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    pdinf, onnx, engine, xml, lite = flags  # export booleans
    file = Path(weights)  # PaddlePaddle weights

    # Load Paddlevvvvv model
    device = select_device(device)
    if half:
        assert device._type() != 1, '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = paddle.zeros([batch_size, 3, *imgsz], dtype="float16" if half else "float32")  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_sublayers():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True
    
    dry_run_model = deepcopy(model)
    for _ in range(2):
        y = dry_run_model(im.astype("float32"))  # dry runs
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PaddlePaddle:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * len(fmts)  # exported filenames
    if pdinf:  # PaddleInference
        f[0], _ = export_paddleinfer(model, im, file)
    if engine: # TensorRT required before ONNX
        f[1], _ = export_engine(model, im, file, half, dynamic, workspace, verbose)
    if onnx:   # ONNX
        f[2], _ = export_onnx(model, im, file, opset, half, dynamic)
    if xml:    # OpenVINO
        f[3], _ = export_openvino(model, im, file, metadata, half)
    if lite:   # PaddleLite
        f[4], _ = export_paddlelite(model, im, file, place, mtype)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # type
        det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
        dir = Path('segment' if seg else 'classify' if cls else '')
        h = '--half' if half else ''  # --half FP16 inference arg
        s = '# WARNING ⚠️ ClassificationModel not yet supported for Paddle Hub AutoShape inference' if cls else \
            '# WARNING ⚠️ SegmentationModel not yet supported for Paddle Hub AutoShape inference' if seg else ''
        LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                    f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                    f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
                    f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
                    f"\nPaddle Hub:      not yet implemented"
                    f'\nVisualize:       https://netron.app')
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pdparams', help='model.pdparams path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--place', default='x86', help='arm, x86, opencl, npu https://www.paddlepaddle.org.cn/lite/v2.12/api_reference/python_api/opt.html')
    parser.add_argument('--mtype', default='naive_buffer', help='naive_buffer, protobuf https://www.paddlepaddle.org.cn/lite/v2.12/api_reference/python_api/opt.html')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TensorRT: dynamic axes')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument(
        '--include',
        nargs='+',
        default=['paddleinfer'],
        help='paddleinfer, onnx, engine, openvino')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
