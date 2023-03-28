# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pdparams --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m paddle.distributed.run --nproc_per_node 4 --master_port 2022 classify/train.py --model yolov5s-cls.pdparams --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pdparams, yolov5s-cls.pdparams, yolov5m-cls.pdparams, yolov5l-cls.pdparams, yolov5x-cls.pdparams
paddle.vision models: --model resnet50, etc. See https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

import paddle
import paddle.distributed as dist
import paddle.optimizer.lr as lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader
from utils.general import (DATASETS_DIR, LOGGER, TQDM_BAR_FORMAT, WorkingDirectory, check_git_info, check_git_status,
                           check_requirements, colorstr, download, increment_path, init_seeds, print_args, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.paddle_utils import (ModelEMA, de_parallel, model_info, reshape_classifier_output, select_device, model_half,
                               paddle_distributed_zero_first, smart_optimizer, smartCrossEntropyLoss, clip_grad_norm_)

GIT_INFO = check_git_info()


def train(opt, device):
    RANK = int(os.getenv('PADDLE_TRAINER_ID', -1)) # RANK is -1 when single GPU or CUP
    WORLD_SIZE = int(os.getenv('PADDLE_TRAINERS_NUM', 1))

    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = isinstance(device, (bool, list, paddle.CUDAPlace))

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pdparams', wdir / 'best.pdparams'

    # Save run settings
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Download Dataset
    with paddle_distributed_zero_first(RANK), WorkingDirectory(ROOT):
        data_dir = data if data.is_dir() else (DATASETS_DIR / data)
        if not data_dir.is_dir():
            LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
            t = time.time()
            if str(data) == 'imagenet':
                subprocess.run(['bash', str(ROOT / 'data/scripts/get_imagenet.sh')], shell=True, check=True)
            else:
                url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Dataloaders
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    print(data_dir / 'train',imgsz,bs // WORLD_SIZE,True,opt.cache,RANK,nw)
    trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                   imgsz=imgsz,
                                                   batch_size=bs // WORLD_SIZE,
                                                   augment=True,
                                                   cache=opt.cache,
                                                   rank=RANK,
                                                   workers=nw)
    test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=bs // WORLD_SIZE * 2,
                                                      augment=False,
                                                      cache=False,
                                                      rank=-1,
                                                      workers=nw)

    # Model
    with paddle_distributed_zero_first(RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith('.pdparams'):
            model = attempt_load(opt.model, fuse=False, verbose=False)
        elif opt.model in paddle.vision.models.__dict__:  # Paddle.Vision models i.e. resnet50
            model = paddle.vision.models.__dict__[opt.model](pretrained=True if pretrained else False)
        else:
            m = ['custom', 'yolov5l', 'yolov5l6', 'yolov5m', 'yolov5m6', 'yolov5n',
                 'yolov5n6', 'yolov5s', 'yolov5s6', 'yolov5x', 'yolov5x6']  # yolov5 models
            raise ModuleNotFoundError(f'--model {opt.model} not found. Available models are: \n' + '\n'.join(m))
        if isinstance(model, DetectionModel):
            LOGGER.warning("WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pdparams'")
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 9)  # convert to classification model
        reshape_classifier_output(model, nc)  # update class count
    for m in model.sublayers():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, paddle.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
    for p in model.parameters():
        p.stop_gradient = False  # for training
    # Info
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes  # attach class names
        model.transforms = testloader.dataset.paddle_transforms  # attach inference transforms
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:25], labels[:25], names=model.names, f=save_dir / 'train_images.jpg')
        logger.log_images(file, name='Train Examples')
        logger.log_graph(model, imgsz)  # log model

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaDecay(opt.lr0, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    # scheduler = lr_scheduler.OneCycleLR(max_learning_rate=lr0, total_steps=epochs, phase_pct=0.1,
    #                                    divide_factor=1 / 25 / lrf)

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, scheduler, momentum=0.9, decay=opt.decay)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DP mode
    if isinstance(device, List) and paddle.device.cuda.device_count() > 1:
        LOGGER.info('Using Multi-GPU training, use paddle.distributed.launch for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/update_cn.html#danjiduokaqidong.')
        model = paddle.DataParallel(model)

    # Train
    t0 = time.time()
    criterion = smartCrossEntropyLoss(classes=nc, label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = paddle.amp.GradScaler(enable=cuda)
    val = test_dir.stem  # 'val' or 'test'
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        if RANK != -1:
            trainloader.batch_sampler.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:  # progress bar
            # Forward
            with paddle.amp.auto_cast(cuda):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (paddle.device.cuda.memory_reserved() / 1E9 if paddle.device.is_compiled_with_cuda() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate.run(model=ema.ema,
                                                     dataloader=testloader,
                                                     criterion=criterion,
                                                     pbar=pbar)  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy

        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                'train/loss': tloss,
                f'{val}/loss': vloss,
                'metrics/accuracy_top1': top1,
                'metrics/accuracy_top5': top5,
                'lr/0': optimizer.get_lr()}  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': model_half(deepcopy(ema.ema)),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                paddle.save(ckpt, str(last))
                if best_fitness == fitness:
                    paddle.save(ckpt, str(best))
                del ckpt

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                    f"\nResults saved to {colorstr('bold', save_dir)}"
                    f'\nPredict:         python classify/predict.py --weights {best} --source im.jpg'
                    f'\nValidate:        python classify/val.py --weights {best} --data {data_dir}'
                    f'\nExport:          python export.py --weights {best} --include onnx'
                    f"\nPaddle Hub:      not yet implemented"
                    f'\nVisualize:       https://netron.app\n')

        # Plot examples
        images, labels = (x[:25] for x in next(iter(testloader)))  # first 25 images and labels
        pred = paddle.argmax(ema.ema(images), 1)
        file = imshow_cls(images, labels, pred, de_parallel(model).names, verbose=False, f=save_dir / 'test_images.jpg')

        # Log results
        meta = {'epochs': epochs, 'top1_acc': best_fitness, 'date': datetime.now().isoformat()}
        logger.log_images(file, name='Test Examples (true-predicted)', epoch=epoch)
        logger.log_model(best, epochs, metadata=meta)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s-cls.pdparams', help='initial weights path')
    parser.add_argument('--data', type=str, default='imagenette160', help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True, help='start from i.e. --pretrained False')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    RANK = int(os.getenv('PADDLE_TRAINER_ID', -1))
    WORLD_SIZE = int(os.getenv('PADDLE_TRAINERS_NUM', 1))

    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert paddle.device.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
        dist.init_parallel_env()

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)


def run(**kwargs):
    # Usage: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
