# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/_image.py
# Training / validation data based on images of cortex.

#===============================================================================
# Dependencies

#-------------------------------------------------------------------------------
# External Libries
import os, sys, time, copy, pimms, PIL, warnings, torch
import numpy as np
import scipy as sp
import nibabel as nib
import pyrsistent as pyr
import neuropythy as ny
import matplotlib as mpl
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Internal Tools

# commented out below two / three lines: unnecessary for current work
# from .config import (default_partition, default_image_size, saved_image_size)
# from .util import (sids, partition_id, partition as partition_sids,
#                   is_partition, trndata, valdata, convrelu)

from .util import convrelu

from torch.utils.data import (Dataset, DataLoader)
from torchvision import (transforms, models)

#-------------------------------------------------------------------------------
# Image-based CNN Model Code

class UNet(torch.nn.Module):
    """a U-Net with a ResNet18 backbone for learning visual area labels.

    The `UNet` class implements a ["U-Net"](https://arxiv.org/abs/1505.04597)
    with a [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) bacbone.
    The class inherits from `torch.nn.Module`.
    
    The original implementation of this class was by Shaoling Chen
    (sc6995@nyu.edu), and additional modifications have been made by Noah C.
    Benson (nben@uw.edu).

    Parameters
    ----------
    feature_count : int
        The number of channels (features) in the input image. When using an
        `HCPVisualDataset` object for training, this value should be set to 4
        if the dataset uses the `'anat'` or `'func'` features and 8 if it uses
        the `'both'` features.
    class_count : int
        The number of output classes in the label data. For V1-V3 this is
        typically either 3 (V1, V2, V3) or 6 (LV1, LV2, LV3, RV1, RV2, RV3).
    pretrained_resnet : boolean, optional
        Whether to use a pretrained resnet for the backbone (default: False).
    middle_branches : boolean, optional
        Whether to include a set of branched filters in the middle of the
        `UNet`. These filters can improve the model's performance in some cases.
        The default is `False`.
    apply_sigmoid : boolean, optional
        Whether to apply the sigmoid function to the outputs. The default is
        `False`.

    Attributes
    ----------
    pretrained_resnet : boolean
        `True` if the resnet used in this `UNet` was originally pre-trained and
        `False` otherwise.
    base_model : PyTorch Module
        The ResNet-18 model that is used as the backbone of the `UNet` model.
    base_layers : list of PyTorch Modules
        The ResNet-18 layers that are used in the backbone of the `UNet` model.
    feature_count : int
        The number of input channels (features) that the model expects in input
        images.
    class_count : int
        The number of classes predicted by the model.
    middle_branches : int
        The number of middle-branches used in the model.
    """
    def __init__(self, feature_count, class_count,
                 pretrained_resnet=False,
                 middle_branches=False,
                 apply_sigmoid=False):
        import torch.nn as nn
        # Initialize the super-class.
        super().__init__()
        # Store some basic attributes.
        self.feature_count = feature_count
        self.class_count = class_count
        self.pretrained_resnet = pretrained_resnet
        self.apply_sigmoid = apply_sigmoid
        # Set up the base model and base layers for the model.
        self.base_model = models.resnet18(pretrained=pretrained_resnet)
        if feature_count != 3:
            # Adjust the first convolution's number of input channels.
            c1 = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                feature_count, c1.out_channels,
                kernel_size=c1.kernel_size, stride=c1.stride,
                padding=c1.padding, bias=c1.bias)
        self.base_layers = list(self.base_model.children())
        # Make the U-Net layers out of the base-layers.
        # size = (N, 64, H/2, W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size = (N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        # size = (N, 128, H/8, W/8)        
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        # size = (N, 256, H/16, W/16)
        self.layer3 = self.base_layers[6]  
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        # size = (N, 512, H/32, W/32)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        # In the middle of the UNet, we add some branched middle layers.
        if middle_branches is True:
            middle_branches = 3
        elif middle_branches is False or middle_branches is None:
            middle_branches = 0
        self.middle_branches = middle_branches
        if middle_branches > 0:
            if middle_branches == 4:
                def _branch(n_ignored):
                    return nn.Sequential(
                        convrelu(512, 256, 3, 1),
                        convrelu(256, 128, 1, 0))
            elif middle_branches == 3:
                def _branch(n_out):
                    return nn.Sequential(
                        convrelu(512, 341, 3, 1),
                        convrelu(341, n_out, 1, 0))
            else:
                raise ValueError("only 3 or 4 middle branches supported")
            # Note that if branches == 4, the arg to _branch is ignored.
            self.middle_branch1 = _branch(170)
            self.middle_branch2 = _branch(170)
            self.middle_branch2 = _branch(171)
            self.middle_branch4 = None if middle_branches == 3 else _branch()
            self.middle_converge = nn.Sequential(
                convrelu(512, 512, 3, 1),
                convrelu(512, 512, 1, 0))
        else:
            self.middle_branch1 = None
            self.middle_branch2 = None
            self.middle_branch3 = None
            self.middle_branch4 = None
            self.middle_converge = None
        # The up-swing of the UNet; we will need to upsample the image.
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size0 = convrelu(feature_count, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, class_count, 1)


    def forward(self, input):
        # Do the original size convolutions.
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # Now the front few layers, which we save for adding back in on the UNet
        # up-swing below.
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        # If there are middle-branches, we run them here.
        if self.middle_branches == 0:
            midx = layer4
        else:
            if self.middle_branches == 4:
                midx = torch.cat([self.middle_branch1(layer4),
                                  self.middle_branch2(layer4),
                                  self.middle_branch3(layer4),
                                  self.middle_branch4(layer4)],
                                 dim=1)
            elif self.middle_branches == 3:
                midx = torch.cat([self.middle_branch1(layer4),
                                  self.middle_branch2(layer4),
                                  self.middle_branch3(layer4)],
                                 dim=1)
            midx = self.middle_converge(midx)
        # Now, we start the up-swing; each step must upsample the image.
        layer4 = self.layer4_1x1(midx)
        # Up-swing Step 1
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        # Up-swing Step 2
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        # Up-swing Step 3
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        # Up-swing Step 4
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        # Up-swing Step 5
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        # And the final convolution.
        out = self.conv_last(x)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out
