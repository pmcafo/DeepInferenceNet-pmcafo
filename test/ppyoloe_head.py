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
                param_group_conv_weight['base_lr']