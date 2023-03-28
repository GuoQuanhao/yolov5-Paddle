import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..general import xywh2xyxy
from ..loss import FocalLoss, smooth_BCE
from ..metrics import bbox_iou
from ..paddle_utils import de_parallel
from .general import crop_mask


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, overlap=False):
        self.sort_obj_iou = False
        self.overlap = overlap
        h = de_parallel(model).hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor([h['cls_pw']]))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor([h['obj_pw']]))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.nm = m.nm  # number of masks
        self.anchors = m.anchors

    def __call__(self, preds, targets, masks):  # predictions, targets, model
        p, proto = preds
        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        lcls = paddle.zeros([1])
        lbox = paddle.zeros([1])
        lobj = paddle.zeros([1])
        lseg = paddle.zeros([1])
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, targets)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = paddle.zeros(pi.shape[:4], dtype=pi.dtype)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1, self.nc, nm), 1)  # subset of predictions

                # Box regression
                pxy = F.sigmoid(pxy) * 2 - 0.5
                pwh = (F.sigmoid(pwh) * 2) ** 2 * anchors[i]
                pbox = paddle.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clip(0).astype(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = paddle.full_like(pcls, self.cn)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls.astype("float32"), t.astype("float32"))  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]
                marea = xywhn[i][:, 2:].prod(1)  # mask width, height normalized
                mxyxy = xywh2xyxy(xywhn[i] * paddle.to_tensor([mask_w, mask_h, mask_w, mask_h]))
                for bi in b.unique():
                    j = b == bi  # matching index
                    if self.overlap:
                        mask_gti = paddle.where(masks[bi][None] == tidxs[i][j].reshape([-1, 1, 1]), 1.0, 0.0).astype("float32")
                    else:
                        mask_gti = masks[tidxs[i]][j].astype("float32")
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])

            obji = self.BCEobj(pi[..., 4].astype("float32"), tobj.astype("float32"))
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lseg *= self.hyp['box'] / bs

        loss = lbox + lobj + lcls + lseg
        return loss * bs, paddle.concat((lbox, lseg, lobj, lcls)).detach()

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.reshape([self.nm, -1])).reshape([-1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask.astype("float32"), gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(axis=(1, 2)) / area).mean()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
        gain = paddle.ones([8])  # normalized to gridspace gain
        ai = paddle.arange(na, dtype="float32").reshape([na, 1]).tile([1, nt])  # same as .repeat_interleave(nt)
        if self.overlap:
            batch = p[0].shape[0]
            ti = []
            for i in range(batch):
                num = (targets[:, 0] == i).sum()  # find number of targets of each image
                if num.item() == 0:
                    continue
                ti.append(paddle.arange(num, dtype="float32").reshape([1, num]).tile([na, 1]) + 1)  # (na, num)
            ti = paddle.concat(ti, 1)  # (na, nt)
        else:
            ti = paddle.arange(nt, dtype="float32").reshape([1, nt]).tile([na, 1])
        targets = paddle.concat((targets.tile([na, 1, 1]), ai[..., None], ti[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = paddle.to_tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            dtype="float32") * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = gain[2:6] * 0 + paddle.to_tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = paddle.maximum(r, 1 / r).max(2) < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = paddle.concat((paddle.ones_like(j).unsqueeze(0), j.unsqueeze(0), k.unsqueeze(0), l.unsqueeze(0), m.unsqueeze(0)))
                t = t.tile((5, 1, 1))[j]
                offsets = (paddle.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, at = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            (a, tidx), (b, c) = at.astype(paddle.int64).T, bc.astype(paddle.int64).T  # anchors, image, class
            gij = (gxy - offsets).astype(paddle.int64)
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clip_(0, shape[2] - 1), gi.clip_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(paddle.concat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tidxs.append(tidx)
            xywhn.append(paddle.concat((gxy, gwh), 1) / gain[2:6])  # xywh normalized

        return tcls, tbox, indices, anch, tidxs, xywhn
