# YOLOv5 reproduction 🚀 by GuoQuanhao


import paddle


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen

    Returns:
        YOLOv5 paddle model
    """
    from pathlib import Path

    from models.yolo import Model
    from utils.general import check_requirements, set_logging
    from utils.downloads import attempt_download
    from utils.paddle_utils import select_device

    file = Path(__file__).resolve()
    check_requirements(exclude=('visualdl', 'thop', 'opencv-python'))
    set_logging(verbose=verbose)

    save_dir = Path('') if str(name).endswith('.pdparams') else file.parent
    path = (save_dir / name).with_suffix('.pdparams')  # checkpoint path
    try:
        cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml path
        model = Model(cfg, channels, classes)  # create model
        if pretrained:
            ckpd = paddle.load(attempt_download(path))  # load
            msd = model.state_dict()  # model state_dict
            csd = ckpd['state_dict']  # checkpoint state_dict
            csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
            model.set_state_dict(csd)  # load
            if len(ckpd['model'].names) == classes:
                model.names = ckpd['model'].names  # set class names attribute
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model

    except Exception as e:
        help_url = 'https://github.com/GuoQuanhao/YOLOv5-Paddle'
        s = 'Make a issue at %s for help.' % help_url
        raise Exception(s) from e


def custom(path='path/to/model.pdparams', autoshape=True, verbose=True):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=verbose)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-nano model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5n', pretrained, channels, classes, autoshape, verbose)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-small model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5s', pretrained, channels, classes, autoshape, verbose)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-medium model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5m', pretrained, channels, classes, autoshape, verbose)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-large model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5l', pretrained, channels, classes, autoshape, verbose)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-xlarge model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5x', pretrained, channels, classes, autoshape, verbose)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-nano-P6 model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5n6', pretrained, channels, classes, autoshape, verbose)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-small-P6 model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5s6', pretrained, channels, classes, autoshape, verbose)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-medium-P6 model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5m6', pretrained, channels, classes, autoshape, verbose)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-large-P6 model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5l6', pretrained, channels, classes, autoshape, verbose)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    # YOLOv5-xlarge-P6 model https://github.com/GuoQuanhao/YOLOv5-Paddle
    return _create('yolov5x6', pretrained, channels, classes, autoshape, verbose)


if __name__ == '__main__':
    model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # pretrained
    # model = custom(path='path/to/model.pdparams')  # custom

    # Verify inference
    import cv2
    import numpy as np
    from PIL import Image
    from pathlib import Path

    imgs = ['data/images/zidane.jpg',  # filename
            Path('data/images/zidane.jpg'),  # Path
            'https://github.com/GuoQuanhao/YOLOv5-Paddle/data/images/zidane.jpg',  # URI
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
            Image.open('data/images/bus.jpg'),  # PIL
            np.zeros((320, 640, 3))]  # numpy

    results = model(imgs)  # batched inference
    results.print()
    results.save()
