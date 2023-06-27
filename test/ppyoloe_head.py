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

    def forward(self, fe