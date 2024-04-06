import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import cv2
import time

from cspresnet import CSPResNet
from custom_pan import CustomCSPPAN
from ppyoloe_head import PPYOLOEHead
from resnet import ConvNormLayer, BottleNeck, ResNet
from test2_utils import save_as_txt, save_weights_as_txt, save_weights_as_miemienet, load_ckpt


def swish(x):
    return x * torch.sigmoid(x)


class Res(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True):
        super(Res, self).__init__()
        self.fc = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=bias)
        if bias:
            torch.nn.init.normal_(self.fc.bias, 0., 1.)
        # self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.33)
   