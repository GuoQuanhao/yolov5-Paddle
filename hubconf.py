# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license

import paddle


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=''):
    """Creates or loads a YOLOv5 model

    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pdparams'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, paddle.device, ''): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.paddle_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pdparams') if name.suffix == '' and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    if model.pdparams and isinstance(model.model, ClassificationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. '
                                       'You must pass paddle tensors in BCHW to this model, i.e. shape(1,3,224,224).')
                    elif model.pdparams and isinstance(model.model, SegmentationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. '
                                       'You will not be able to run inference with this model.')
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = paddle.load(attempt_download(path))['model']  # load
                csd = state_dict_float(ckpt)# checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.set_state_dict(csd,)  # load
                if len(ckpt['model']['names']) == classes:
                    model.names = ckpt['model']['names']  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model

    except Exception as e:
        help_url = 'https://github.com/GuoQuanhao/yolov5/issues'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


def custom(path='path/to/model.pdparams', autoshape=True, _verbose=True, device=''):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-nano model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5n', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-small model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5s', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-medium model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5m', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-large model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5l', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-xlarge model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5x', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-nano-P6 model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5n6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-small-P6 model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5s6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-medium-P6 model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5m6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-large-P6 model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5l6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=''):
    # YOLOv5-xlarge-P6 model https://github.com/GuoQuanhao/yolov5
    return _create('yolov5x6', pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='model name')
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pdparams')  # custom

    # Images
    imgs = [
        'data/images/zidane.jpg',  # filename
        Path('data/images/zidane.jpg'),  # Path
        'http://182.61.54.236/yolov5/zidane.jpg',  # URI
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
        Image.open('data/images/bus.jpg'),  # PIL
        np.zeros((320, 640, 3))]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
