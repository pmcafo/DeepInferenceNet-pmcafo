import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import cv2
import time



from test2_utils import save_as_txt, save_weights_as_txt, read_weights_from_miemienet, load_ckpt




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.act = nn.LeakyReLU(0.33)

    def __call__(self, x):
        y = self.act(x)
        retu