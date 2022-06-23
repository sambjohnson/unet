# ---- torch imports --- #
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# --- general system imports --- #
from typing import Dict, Any

import pandas as pd
import numpy as np

import os
import sys
import pickle
import struct
import copy
from array import array

# --- image processing imports --- #
import png
from PIL import Image
from PIL import ImageOps

import matplotlib.pyplot as plt
import numpy as np

# --- utility functions --- #


def get_device():
    """ Returns gpu if available, cpu otherwise."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_requires_grad(module, allfalse=False):
    """ Checks whether all of a module's parameters
        have requires_grad = True. Returns False if not.
        If allfalse=True, instead returns True only if
        all module parameters do NOT require grad.
        
        Carefully understand allfalse=True behavior.
    """
    assert isinstance(module, nn.Module)
    rg = [p.requires_grad for p in module.parameters()]
    ret = all(r == (not allfalse) for r in rg)
    return ret


def set_unfreeze_(model, submodules_to_unfreeze):
    """ In-place unfreezes only specified submodules
        given in list (submodules_to_unfreeze);
        freezes all other parts of model.
        Useful for transfer learning with pretrained models
        and doing combined feature extraction / finetuning.
    """
    model.requires_grad_(False)  # freezes entire model
    for subm in submodules_to_unfreeze:
        subm.requires_grad_(True)  # unfreezes just specified sm's
    return


def make_pretrained_state_dict(model_state, pretrained_state):
    """ Helper function to load copy a portion of a model's state
        from another model with (partially) overlapping architecture.
        Args:
            model_state: a PyTorch-style OrderedDict with the target state
                of the model to initialize
            pretrained_state: a PyTorch-style OrderedDict with the loaded
                (presumably pretrained) parameters from a partially comparable
                model.
        Returns:
            warmstart_state: a state_dict that is equal to an updated version
            model_state, where keys that are also present in pretrained_state
            are updated to their values in that state. E.g., pretrained_state
            may be the parameters of a large, pretrained model, and model_state
            may be the state of an architecture that only uses some of that
            model's first layeres.

    """
    pd = pretrained_state
    sd = model_state
    warmstart_params = copy.deepcopy(sd)
    for k, v in pd.items():
        warmstart_params[k] = v
    return warmstart_params


def imshow(inp, title=None, normalize=False, figsize=None):
    """ Imshow for Tensor. Visualizes images in a grid. """
    inp = inp.numpy().transpose((1, 2, 0))
    
    if figsize is None:
        figsize = (20, 10)

    if normalize:
        # normalization may be required in some cases, but not here
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    
    plt.figure(figsize = figsize)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

