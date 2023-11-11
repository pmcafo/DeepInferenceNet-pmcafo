# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import distributed as dist
import torch.nn.functional as F
import numpy as np
import copy

# from ppdet.modeling.layers import MultiClassNMS
# from mmdet.models.assigners.utils import generate_anchors_for_grid_cell
# from mmdet.models.backbones.cspresnet import ConvBNLayer
# from mmdet.models.bbox_utils import batch_distance2bbox
# from mmdet.models.matrix_nms import matrix_nms
# from mmdet.models.ops import get_static_shape, paddle_distributed_is_initialized, get_act_fn
# from mmdet.models.initializer import bias_init_with_prob, constant_, normal_
# from mmdet.models.losses.iou_losses import GIoULoss
# from mmdet.utils import my_multiclass_nms, get_world_size
from cspresnet import ConvBNLayer, BasicBlock
from cspresnet import get_act_fn
import ncnn_utils as ncnn_utils



def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox



def print_diff(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        aaaaaa1 = dic[key]
        aaaaaa2 = tensor.cpu().detach().numpy()
        ddd = np.sum((aaaaaa1 - aaaaaa2) ** 2)
        print('diff=%.6f (%s)' % (ddd, key))


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish', act_name='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act, act_name=act_name)

        # self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)

    def export_ncnn(self, ncnn_data, bottom_names):
        feat = bottom_names[0]
        avg_feat = bottom_names[1]

        weight = ncnn_utils.conv2d(ncnn_data, [avg_feat, ], self.fc, 'sigmoid')

        # 然后是逐元素相乘
        bottom_names = ncnn_utils.binaryOp(ncnn_data, [feat, weight[0]], op='Mul')
        bottom_names = self.conv.export_ncnn(ncnn_data, bottom_names)
        return bottom_names

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.fc.weight.requires_grad:
            param_group_conv_weight = {'params': [self.fc.weight]}
            param_group_conv_weight['lr'] = base_lr * 1.0
            param_group_conv_weight['base_lr'] = base_lr * 1.0
            param_group_conv_weight['weight_decay'] = base_wd
            param_group_conv_weight['need_clip'] = need_clip
            param_group_conv_weight['clip_norm'] = clip_norm
            param_groups.append(param_group_conv_weight)
        if self.fc.bias.requires_grad:
            param_group_conv_bias = {'params': [self.fc.bias]}
            param_group_conv_bias['lr'] = base_lr * 1.0
            param_group_conv_bias['base_lr'] = base_lr * 1.0
            param_group_conv_bias['weight_decay'] = base_wd
            param_group_conv_bias['need_clip'] = need_clip
            param_group_conv_bias['clip_norm'] = clip_norm
            param_groups.append(param_group_conv_bias)


class PPYOLOEHead(nn.Module):
    __shared__ = ['num_classes', 'eval_size', 'trt', 'exclude_nms']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 nms_cfg=None,
                 exclude_nms=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        # self.iou_loss = GIoULoss()
        self.iou_loss = None
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        # if isinstance(self.nms, MultiClassNMS) and trt:
        #     self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.nms_cfg = nms_cfg
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act_name = act
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act, act_name=act_name))
            self.stem_reg.append(ESEAttn(in_c, act=act, act_name=act_name))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(len(self.in_channels)):
            self.stem_cls[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.stem_reg[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            if self.pred_cls[i].weight.requires_grad:
                param_group_conv_weight = {'params': [self.pred_cls[i].weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.pred_cls[i].bias.requires_grad:
                param_group_conv_bias = {'params': [self.pred_cls[i].bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)
            if self.pred_reg[i].weight.requires_grad:
                param_group_conv_weight2 = {'params': [self.pred_reg[i].weight]}
                param_group_conv_weight2['lr'] = base_lr * 1.0
                param_group_conv_weight2['base_lr'] = base_lr * 1.0
                param_group_conv_weight2['weight_decay'] = base_wd
                param_group_conv_weight2['need_clip'] = need_clip
                param_group_conv_weight2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight2)
            if self.pred_reg[i].bias.requires_grad:
                param_group_conv_bias2 = {'params': [self.pred_reg[i].bias]}
                param_group_conv_bias2['lr'] = base_lr * 1.0
                param_group_conv_bias2['base_lr'] = base_lr * 1.0
                param_group_conv_bias2['weight_decay'] = base_wd
                param_group_conv_bias2['need_clip'] = need_clip
                param_group_conv_bias2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        # bias_cls = bias_init_with_prob(0.01)
        # for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
        #     constant_(cls_.weight)
        #     constant_(cls_.bias, bias_cls)
        #     constant_(reg_.weight)
        #     constant_(reg_.bias, 1.0)

        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, 1)
        reg_distri_list = torch.cat(reg_distri_list, 1)

        # import numpy as np
        # dic = np.load('../aaa.npz')
        # cls_score_list = torch.Tensor(dic['cls_score_list'])
        # reg_distri_list = torch.Tensor(dic['reg_distri_list'])
        # anchors = torch.Tensor(dic['anchors'])
        # anchor_points = torch.Tensor(dic['anchor_points'])
        # stride_tensor = torch.Tensor(dic['stride_tensor'])
        # gt_class = torch.Tensor(dic['gt_class'])
        # gt_bbox = torch.Tensor(dic['gt_bbox'])
        # pad_gt_mask = torch.Tensor(dic['pad_gt_mask'])
        # targets['gt_class'] = gt_class
        # targets['gt_bbox'] = gt_bbox
        # targets['pad_gt_mask'] = pad_gt_mask
        #
        # loss = torch.Tensor(dic['loss'])
        # loss_cls = torch.Tensor(dic['loss_cls'])
        # loss_iou = torch.Tensor(dic['loss_iou'])
        # loss_dfl = torch.Tensor(dic['loss_dfl'])
        # loss_l1 = torch.Tensor(dic['loss_l1'])

        losses = self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)
        return losses

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], -1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    [h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            ttttttttttttttt = self.stem_cls[i](feat, avg_feat) + feat
            aaaaaaaa = ttttttttttttttt.permute((0, 2, 3, 1)).cpu().detach().numpy()
            cls_logit = self.pred_cls[i](ttttttttttttttt)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l])
            reg_dist = reg_dist.permute((0, 2, 1, 3))
            reg_dist = F.softmax(reg_dist, dim=1)
            aaaaaaaa = reg_dist.permute((0, 2, 3, 1)).cpu().detach().numpy()
            aaaaaaaabb = reg_dist.permute((0, 3, 2, 1)).cpu().detach().numpy()
            reg_dist = self.proj_conv(reg_dist)
            aaaaaaaabb222 = reg_dist.permute((0, 3, 2, 1)).cpu().detach().numpy()
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = torch.cat(reg_dist_list, -1)    # [N,  4, A]

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def export_ncnn(self, ncnn_data, bottom_names):
        feats = bottom_names
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
   