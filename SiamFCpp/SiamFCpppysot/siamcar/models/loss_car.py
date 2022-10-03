"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F

INF = 100000000

def get_cls_loss(pred, label, select):

    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    # https://blog.csdn.net/g_blink/article/details/102854188
    #  torch.index_select(args1, args2, tensor) ，第一个参数：索引对象， 第二个参数：索引方式 按照行索引（0）or 按照列索引（1），第三个参数：索引的序号 
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
 
    # https://blog.csdn.net/qq_22210253/article/details/85229988
    # NLLLoss
    # CrossEntropyLoss 描述两个概率分布之间的距离，交叉熵结果越小说明二者越接近，就是把Softmax-Log-NLLLoss合并成一步； BCELoss是二分类交叉熵损失 
    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):  # pred=[32, 1, 25, 25, 2]
    pred = pred.view(-1, 2) #[32,1,25, 25, 2] --> [20000, 2]
    label = label.view(-1) #-->[20000]

    pos = label.data.eq(1).nonzero().squeeze().cuda() # 获取label中pos索引
    neg = label.data.eq(0).nonzero().squeeze().cuda() # 获取label中neg索引
    loss_pos = get_cls_loss(pred, label, pos) #pred[20000,2], label[20000], pos，正样本索引
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None, eps=1e-7):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        b1_x1, b1_x2 = pred_left - pred_right / 2, pred_left + pred_right / 2
        b1_y1, b1_y2 = pred_top - pred_bottom / 2, pred_top + pred_bottom / 2
        b2_x1, b2_x2 = target_left - target_right / 2, target_left + target_right / 2
        b2_y1, b2_y2 = target_top - target_bottom / 2, target_top + target_bottom / 2

        # pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        # target_area = (target_left + target_right) * (target_top + target_bottom)
        #
        # w1 = target_left + target_right
        # h1 = target_top + target_bottom
        # w2 = pred_left + pred_right
        # h2 = pred_top + pred_bottom
        #
        # w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        # g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        # h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        # g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        # ac_uion = g_w_intersect * g_h_intersect + 1e-7
        # area_intersect = w_intersect * h_intersect
        # area_union = target_area + pred_area - area_intersect
        # ious = (area_intersect + 1.0) / (area_union + 1.0)
        # gious = ious - (ac_uion - area_union) / ac_uion
        #
        # if self.loc_loss_type == 'iou':
        #     losses = -torch.log(ious)
        # elif self.loc_loss_type == 'linear_iou':
        #     losses = 1 - ious
        # elif self.loc_loss_type == 'giou':
        #     losses = 1 - gious
        # elif self.loc_loss_type == 'CIoU' or self.loc_loss_type == 'DIoU':  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        #     c2 = g_w_intersect ** 2 + g_h_intersect ** 2 + 1e-7  # convex diagonal squared
        #     rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
        #             (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
        #     if self.loc_loss_type == 'DIoU':
        #         losses = 1 - ious + rho2 / c2  # DIoU
        #     elif self.loc_loss_type == 'CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        #         v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        #         with torch.no_grad():
        #             alpha = v / (v - ious + (1 + 1e-7))
        #         losses = 1 - ious + (rho2 / c2 + v * alpha)  # CIoU
        #
        # else:
        #     raise NotImplementedError

        # Intersection area   tensor.clamp(0): 将矩阵中小于0的元数变成0
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if self.loc_loss_type == 'GIoU' or self.loc_loss_type == 'DIoU' or self.loc_loss_type == 'CIoU':
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 两个框的最小闭包区域的width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 两个框的最小闭包区域的height
            if self.loc_loss_type == 'CIoU' or self.loc_loss_type == 'DIoU':  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if self.loc_loss_type == 'DIoU':
                    losses = 1 - iou + rho2 / c2  # DIoU
                elif self.loc_loss_type == 'CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    losses = 1 - iou + (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                losses = 1 - iou + (c_area - union) / c_area  # GIoU
        else:
            losses = 1 - iou  # IoU

        # weight = None
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """
    def __init__(self, cfg):

        #(1) 回归分支 IOU损失
        self.box_reg_loss_func = IOULoss(loc_loss_type='CIoU')

        #(2) 中心分支 二元交叉熵损失函数 https://blog.csdn.net/qq_22210253/article/details/85222093
        # BCELoss： 1/n *Sigma(Yn*ln(Xn)+(1-Yn)*ln(1-Xn)) ，其中Y是target，X是模型输出的值 ； BCELoss的输入需要进行sigmoid处理过的值
        # BCEWithLogitsLoss=Sigmoid+BCELoss 合成一步
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2,-1) #[B,OUTPUT_SIZE,OUTPUT_SIZE] --> [OUTPUT_SIZE**2, B]
        # debug
        #d=xs[:][None]  # [625] --> [1,625]
        #a=xs[:, None]  # [625] --> [625,1]
        # b=bboxes[:, 0] #  [32]
        # c=bboxes[:, 0][None].float() #[1,32]
        
        l = xs[:, None] - bboxes[:, 0][None].float() # [625,1] - [1,32] = [625,32]  网格中心点到目标左边界的距离，
        t = ys[:, None] - bboxes[:, 1][None].float() #          网格中心点到目标上边界的距离，
        r = bboxes[:, 2][None].float() - xs[:, None] #          网格中心点到目标右边界的距离
        b = bboxes[:, 3][None].float() - ys[:, None] #          网格中心点到目标下边界的距离

        reg_targets_per_im = torch.stack([l, t, r, b], dim=2) # [625, 32, 4]

        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float() #[625, 32] 
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float() #[625, 32]
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float() #[625, 32]
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float() #[625, 32]
        is_in_boxes = s1*s2*s3*s4 # [625,32]
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1  #[625,32] ; 0.4×0.4= 16% 目标中心16%的范围的点设置为正样本

        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous()

    def compute_centerness_targets(self, reg_targets): #[xxx, 4]
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]

        # a=left_right.min(dim=-1) # left_right.min(0)返回每一列最小值组成的一维数组; left_right.min(1)返回每一行最小值组成的数组
        # b=left_right.min(dim=1)  # 返回
        # c=left_right.min(dim=-1)[0]

        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets) #获取gt cls和gt_reg [32, 625] ,[32, 625,4],把通道调整到最前面
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4)) #[32,4,25,25] --> [32,25,25,4] --> [32*25*25,4]=[20000,4]
        labels_flatten = (label_cls.view(-1)) #[32, 625]-->[20000]
        reg_targets_flatten = (reg_targets.view(-1, 4)) # [32, 625, 4] --> [20000, 4]
        centerness_flatten = (centerness.view(-1)) # [32, 1, 25, 25] --> [20000]
        # torch.nonzero() 返回一个包含输入input中非零元素索引的张量， NxD  （D维张量 输入有N个非0元素，则返回[N,D]的矩阵来表示索引）
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)  # [20000] --> 获取正样本索引

        box_regression_flatten = box_regression_flatten[pos_inds] # pre-bbox
        reg_targets_flatten = reg_targets_flatten[pos_inds]       # reg-bbox 
        centerness_flatten = centerness_flatten[pos_inds]
        # （1）cls loss
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)# 交叉熵损失 nll_loss

        if pos_inds.numel() > 0:

            # center ness label  nll_loss多分类交叉熵损失  https://blog.csdn.net/qq_22210253/article/details/85229988
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # （2） reg loss IOULoss  https://www.pianshen.com/article/72871676564/
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            # （3）center loss  BCELoss 二分类交叉熵损失  https://blog.csdn.net/qq_22210253/article/details/85222093
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss

def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
