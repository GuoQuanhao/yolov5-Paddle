# YOLOv5 reproduction 🚀 by GuoQuanhao
"""
Loss functions
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.metrics import bbox_iou
from utils.paddle_utils import is_parallel
import numpy as np


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Layer):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = F.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - paddle.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Layer):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = paddle.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = F.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Layer):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = F.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = paddle.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor([h['cls_pw']]))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor([h['obj_pw']]))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model._layer.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        lcls, lbox, lobj = paddle.zeros([1]), paddle.zeros([1]), paddle.zeros([1])
        tcls, tbox, indices, anchors = self.build_targets(p, targets.numpy())  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = paddle.zeros_like(pi[:,:,:,:, 0])  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                ps = pi.gather_nd(paddle.concat([b.reshape([-1, 1]), a.reshape([-1, 1]), gj.reshape([-1, 1]), gi.reshape([-1, 1])], 1))
                # Regression
                pxy = F.sigmoid(ps[:, :2]) * 2. - 0.5
                pwh = (F.sigmoid(ps[:, 2:4]) * 2) ** 2 * anchors[i]
                pbox = paddle.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) # 设置梯度
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clip(0).astype(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = paddle.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                with paddle.no_grad():
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = paddle.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in paddle.concat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[:,:,:,:, 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, paddle.concat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.tile(np.arange(na, dtype=np.float32).reshape(na, 1), [1, nt])  # same as .repeat_interleave(nt)
        targets = np.concatenate((np.tile(targets, [na, 1, 1]), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = np.array([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], dtype=np.float32) * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i].numpy()
            gain[2:6] = np.array(p[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = np.maximum(r, 1 / r).max(2) < self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = np.stack((np.ones_like(j), j, k, l, m))
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1) # add line make result same as clamp_
            indices.append((paddle.to_tensor(b),
                            paddle.to_tensor(a),
                            paddle.to_tensor(gj, dtype=paddle.int64),
                            paddle.to_tensor(gi, dtype=paddle.int64)))  # image, anchor, grid indices
            tbox.append(paddle.to_tensor(np.concatenate((gxy - gij, gwh), 1), dtype=paddle.float32))  # box
            anch.append(paddle.to_tensor(anchors[a]))  # anchors
            tcls.append(paddle.to_tensor(c))  # class

        return tcls, tbox, indices, anch