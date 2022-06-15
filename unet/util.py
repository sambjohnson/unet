# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/util.py
# Utilities fo the visual_autolabel library.

"""
The `visual_autolabel.util` package contains utilities for use in and with the
`visual_autolabel` library.
"""

#===============================================================================
# Constants / Globals

# Global Config Items.
# from .config import (default_partition, sids)  # commented out, as unnecessary.


#===============================================================================
# Utility Functions

#-------------------------------------------------------------------------------
# Subject Partitions
# Code for dealing with partitions of training and validation subjects.

def _tensor_to_number(t):
    """Returns the raw numerical data stored in a torch tensor."""
    return t.cpu().numpy()


def is_partition(obj):
    """Returns true if the given object is a subject partition, otherwise False.

    `is_partition(x)` returns `True` if `x` is a mapping with the keys `'trn'`
    and `'val'` or is a tuple with 2 elements. Otherwise, returns `False`.

    Parameters
    ----------
    obj : object
        The object whose quality as a subject partition is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` represents a subject partition and `False` otherwise.
    """
    from collections.abc import Mapping
    return ((isinstance(obj, tuple) and len(obj) == 2) or
            (isinstance(obj, Mapping) and 'trn' in obj and 'val' in obj))


def trndata(obj):
    """Returns the training data of an object representing a subject partition.

    `trndata((trn_data, val_data))` returns `trn_data` (i.e., if given a tuple
    of length 2, `trndata` will return the first element).

    `trndata({'trn': trn_data, 'val': val_data})` also returns `trn_data`.

    See also: `valdata`

    Parameters
    ----------
    obj : mapping or tuple
        Either a dict-like object with the keys `'trn'` and `'val'` or a tuple
        with two elements `(trn, val)`.

    Returns
    -------
    object
        Either the first element of `obj` when `obj` is a tuple or `obj['trn']`
        when `obj` is a mapping.
    """
    if isinstance(obj, tuple):
        return obj[0]
    else:
        return obj['trn']


def valdata(obj):
    """Returns the validation data from a subject partition.

    `valdata((trn_data, val_data))` returns `val_data` (i.e., if given a tuple
    of length 2, `valdata` will return the second element).

    `valdata({'trn': trn_data, 'val': val_data})` also returns `val_data`.

    Parameters
    ----------
    obj : mapping or tuple
        Either a dict-like object with the keys `'trn'` and `'val'` or a tuple
        with two elements `(trn, val)`.

    Returns
    -------
    object
        Either the second element of `obj` when `obj` is a tuple or `obj['val']`
        when `obj` is a mapping.
    """
    if isinstance(obj, tuple):
        return obj[1]
    else:
        return obj['val']


def partition_id(obj):
    """Returns a string that uniquely represents a subject partition.

    Parameters
    ----------
    obj : tuple or mapping of a subject partition
        A mapping that contains the keys `'trn'` and `'val'` or a tuple with two
        elements, `(trn, val)`. Both `trn` and `val` must be either iterables of
        subject-ids, datasets with the attribute `sids`, or dataloaders whose
        datasets have th attribute `sids`.

    Returns
    -------
    str
        A hexadecimal string that uniquely represents the partition implied by
        the `obj` parameter.
    """
    from torch.utils.data import (DataLoader, Dataset)
    trndat = trndata(obj)
    valdat = valdata(obj)
    if isinstance(trndat, DataLoader): trndat = trndat.dataset
    if isinstance(valdat, DataLoader): valdat = valdat.dataset
    if isinstance(trndat, Dataset):    trndat = trndat.sids
    if isinstance(valdat, Dataset):    valdat = valdat.sids
    trn = [(sid,'1') for sid in obj[0]]
    val = [(sid,'0') for sid in obj[1]]
    sids = sorted(trn + val, key=lambda x:x[0])
    pid = int(''.join([x[1] for x in sids]), 2)
    return hex(pid)

# note: modified with dummy sids: not necesary here
# sids = list(range(10))  # dummy sids
# def partition(sids, how=default_partition):
#     """Partitions a list of subject-IDs into a training and validation set.
#  
#     `partition(sids, (frac_trn, frac_val))` returns `(trn_sids, val_sids)` where
#     the fraction `frac_trn` of the `sids` have been randomly placed in the
#     training seet and `frac_val` of the subjects have been placed in the 
#     validation set, randomly. The sum `frac_trn + frac_val` must be between 0
#     and 1.
# 
#     `partition(sids, (num_trn, num_val))` where `num_trn` and `num_val` are both
#     positive integers whose sum is less than or equal to `len(sids)` places
#     exactly the number of subject-IDs, randomly, in each category.
# 
#     partition(sids, idstring)` where `idstring` is a hexadecimal string returned
#     by `partition_id()` reproduces the original partition used to create the
#     string.
# 
#     Parameters
#     ----------
#     sids : list-like
#         A list, tuple, array, or iterable of subject identifiers. The
#         identifiers may be numers or strings, but they must be sortable.
#     how : tuple or str
#         Either a tuple `(trn, val)` containing either the fraction of training
#         and validation set members (`trn + val == 1`) or the (integer) 
#         count of training and validation set members (`trn + val == len(sids)`),
#         or a hexadecimal string created by `partition_id`
# 
#     Returns
#     -------
#     tuple of arrays
#         A tuple `(trn_sids, val_sids)` whose members are numpy arrays of the
#         subject-IDs in the training and validation sets, respectively.
#     """
#     import numpy as np
#     sids = np.asarray(sids)
#     n = len(sids)
#     if isinstance(how, tuple):
#         ntrn = trndata(how)
#         nval = valdata(how)
#         if isinstance(ntrn, float) and isinstance(nval, float):
#             if ntrn < 0 or nval < 0: raise ValueError("trn and val must be > 0")
#             nval = round(nval * n)
#             ntrn = round(ntrn * n)
#             tot = nval + ntrn
#             if tot != n: raise ValueError("partition requires trn + val == 1")
#         elif isinstance(ntrn, int) and isinstance(nval, int):
#             if ntrn < 0 or nval < 0: raise ValueError("trn and val must be > 0")
#             tot = ntrn + nval
#             if tot != n: 
#                 raise ValueError("partition requires trn + val == len(sids)")
#         elif isinstance(ntrn, np.ndarray) and isinstance(nval, np.ndarray):
#             a1 = np.unique(sids)
#             a2 = np.unique(np.concatenate([ntrn, nval]))
#             if np.array_equal(a1, a2) and len(a1) == len(sids):
#                 return (ntrn, nval)
#             else:
#                 raise ValueError("partitions must include all sids")
#         else: raise ValueError("trn and val must both be integers or floats")
#         val_sids = np.random.choice(sids, nval)
#         trn_sids = np.setdiff1d(sids, val_sids)
#     elif isinstance(how, str):
#         sids = np.sort(sids)
#         trn_ii = np.array([1 if s == '1' else 0 for s in '{0:b}'.format(how)],
#                           dtype=np.bool)
#         trn_sids = sids[trn_ii]
#         val_sids = sids[~trn_ii]
#     return (trn_sids, val_sids)

#-------------------------------------------------------------------------------
# Filters and PyTorch Modules
# Code for dealing with PyTorch filters and models.

def kernel_default_padding(kernel_size):
    """Returns an appropriate default padding for a kernel size.

    The returned size is `kernel_size // 2`, which will result in an output
    image the same size as the input image.

    Parameters
    ----------
    kernel_size : int or tuple of ints
        Either an integer kernel size or a tuple of `(rows, cols)`.

    Returns
    -------
    int
        If `kernel_size` is an integer, returns `kernel_size // 2`.
    tuple of ints
        If `kernel_size` is a 2-tuple of integers, returns
        `(kernel_size[0] // 2, kernel_size[1] // 2)`.
    """
    try:
        return (kernel_size[0] // 2, kernel_size[1] // 2)
    except TypeError:
        return kernel_size // 2


def convrelu(in_channels, out_channels,
             kernel=3, padding=None, stride=1, bias=True, inplace=True):
    """Shortcut for creating a PyTorch 2D convolution followed by a ReLU.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the convolution.
    out_channels : int
        The number of output channels in the convolution.
    kernel : int, optional
        The kernel size for the convolution (default: 3).
    padding : int or None, optional
        The padding size for the convolution; if `None` (the default), then
        chooses a padding size that attempts to maintain the image-size.
    stride : int, optional
        The stride to use in the convolution (default: 1).
    bias : boolean, optional
        Whether the convolution has a learnable bias (default: True).
    inplace : boolean, optional
        Whether to perform the ReLU operation in-place (default: True).

    Returns
    -------
    torch.nn.Sequential
        The model of a 2D-convolution followed by a ReLU operation.
    """
    import torch
    if padding is None:
        padding = kernel_default_padding(kernel)
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel,
                        padding=padding, bias=bias),
        torch.nn.ReLU(inplace=inplace))

#-------------------------------------------------------------------------------
# Loss Functions

def is_logits(data):
    """Attempts to guess whether the given PyTorch tensor contains logits.

    If the argument `data` contains only values that are no less than 0 and no
    greater than 1, then `False` is returned; otherwise, `True` is returned.
    """
    if   (data > 1).any(): return True
    elif (data < 0).any(): return True
    else:                  return False


def dice_loss(pred, gold, logits=None, smoothing=1, graph=False, metrics=None):
    """Returns the loss based on the dice coefficient.
    
    `dice_loss(pred, gold)` returns the dice-coefficient loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    smoothing : number, optional
        The smoothing coefficient `s`. The default is `1`.
    metrics : dict or None, optional
        An optional dictionary into which the key `'dice'` should be inserted
        with the dice-loss as the value.

    Returns
    -------
    float
        The dice-coefficient loss of the prediction.
    """
    import torch
    pred = pred.contiguous()
    gold = gold.contiguous()
    if logits is None: logits = is_logits(pred)
    if logits: pred = torch.sigmoid(pred)
    intersection = (pred * gold)
    pred = pred**2
    gold = gold**2
    while len(intersection.shape) > 2:
        intersection = intersection.sum(dim=-1)
        pred = pred.sum(dim=-1)
        gold = gold.sum(dim=-1)
    if smoothing is None: smoothing = 0
    loss = (1 - ((2 * intersection + smoothing) / (pred + gold + smoothing)))
    # Average the loss across classes then take the mean across batch elements.
    loss = loss.mean(dim=1).mean()
    if metrics is not None:
        if 'dice' not in metrics: metrics['dice'] = 0.0
        metrics['dice'] += loss.data.cpu().numpy() * gold.size(0)
    return loss


def bce_loss(pred, gold, logits=None, reweight=False, metrics=None):
    """Returns the loss based on the binary cross entropy.
    
    `bce_loss(pred, gold)` returns the binary cross entropy loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    metrics : dict or None, optional
        An optional dictionary into which the key `'bce'` should be inserted
        with the bce-loss as the value.

    Returns
    -------
    float
        The binary cross entropy loss of the prediction.
    """
    import torch
    if logits is None:
        logits = is_logits(pred)
    if logits:
        f = torch.nn.functional.binary_cross_entropy_with_logits
    else:
        f = torch.nn.functional.binary_cross_entropy
    if reweight:
        n = pred.shape[-1] * pred.shape[-2]
        if len(pred.shape) > 3:  # multiply normalization factor by batch size if data is batched
            n *= pred.shape[0]
        r = 0
        for k in range(pred.shape[1]):
            p = pred[:, [k]]
            t = gold[:, [k]]
            r += f(p, t) * (n - torch.sum(t)) / n
    else:
        r = f(pred, gold)
    if metrics is not None:
        if 'bce' not in metrics: metrics['bce'] = 0.0
        metrics['bce'] += r.data.cpu().numpy() * gold.size(0)
    return r


def loss(pred, gold,
         logits=True,
         bce_weight=0.5,
         smoothing=1,
         reweight=True,
         metrics=None):
    """Returns the weighted sum of dice-coefficient and BCE-based losses.

    `loss(pred, gold)` calculates the loss value between the given prediction
    and gold-standard labels, both of which must be the same shape and whose
    elements should represent probability values.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    bce_weight : float, optional
        The weight to give the BCE-based loss; the weight for the 
        dice-coefficient loss is always `1 - bce_weight`. The default is `0.5`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    smoothing : number, optional
        The smoothing coefficient `s` to use with the dice-coefficient liss.
        The default is `1`.
    metrics : dict or None, optional
        An optional dictionary in which the keys `'bce'`, `'dice'`, and `'loss'`
        should be mapped to floating-point values representing the cumulative
        losses so far across samples in the epoch. The losses of this
        calculation are added to these values.

    Returns
    -------
    number
        The weighted sum of losses of the prediction.
    """
    if bce_weight < 0 or bce_weight > 1:
        raise ValueError("bce_weight must be between 0 and 1")
    else:
        dice_weight = 1 - bce_weight
    if logits is None: logits = is_logits(pred)
    bce = bce_loss(pred, gold,
                   logits=logits,
                   reweight=reweight,
                   metrics=metrics)
    dice = dice_loss(pred, gold,
                     logits=logits,
                     smoothing=smoothing,
                     metrics=metrics)
    loss = bce * bce_weight + dice * dice_weight
    if metrics is not None:
        if 'loss' not in metrics: metrics['loss'] = 0.0
        metrics['loss'] += loss.data.cpu().numpy() * gold.size(0)
    return loss

#===============================================================================
# __all__
__all__ = ["is_partition"
           ,"trndata"
           ,"valdata"
#           ,"partition"
           ,"partition_id"
           ,"kernel_default_padding"
           ,"convrelu"
           ,"is_logits"
           ,"dice_loss"
           ,"bce_loss"
           ,"loss"]
