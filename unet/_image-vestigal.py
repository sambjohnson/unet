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


#===============================================================================
# Dataset and CNN Implementation

#-------------------------------------------------------------------------------
# HCPVisualDataset
# The PyTorch dataset class for the HCP and functions for creating datasets and
# data-loaders 

class HCPVisualDataset(Dataset):
    """A PyTorch dataset for the HCP.

    `HCPVisualDataset` is a PyTorch dataset for the HCP. It produces `4xNx2N`
    images of a subject's occipital poles (LH and RH) in which the four
    channels correspond various data on cortical surface depending on whether
    the dataset is configured for anatomical data, functional data, or both.
    
    For anatomical images, the channels correspond to the following:
     0. curvature (-1:1 is rescaled to 0:1)
     1. sulcal depth (-2:2 is rescaled to 0:1)
     2. cortical thickness (1:6 is rescaled to 0:1)
     3. midgray surface area (0:3 is rescaled to 0:1)

    For functional images, the channels correspond to:
     0. polar angle (-180:180 rescaled to 0:1)
     1. eccentricity (0:12 rescaled to 0:1)
     2. size/sigma (0:4 rescaled to 0:1)
     3. variance explained (not rescaled)

    For both anatomical and functional (`'both'`), 8 layers are returned,
    consisting of the anatomical layers followed by the functional layers.

    Parameters
    ----------
    sids : iterable of subject-IDs
        The subjects to include in this dataset.
    features : 'anat' or 'func' or 'both', optional
        The features to use in the dataset's images.
    image_size : int, optional
        The width of the training images, in pixels (default: 512).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).

    Attributes
    ----------
    sids : numpy array
        A numpy array of integer subject IDs.
    image_size : int
        The width of images being usd in the dataset.
    features : 'anat' or 'func' or 'both'
        The features being used in th dataset.
    cache_path : str or None
        The path in which the data for this dataset is being cached, if any.
    """
    anat_layers = {'curvature': (-1,1), 
                   'convexity':(-2,2),
                   'thickness':(1,6),
                   'surface_area':(0,3)}
    func_layers = {'prf_polar_angle': (-180,180),
                   'prf_eccentricity':(0,12),
                   'prf_radius':(0,4),
                   'prf_variance_explained': (0,1)}
    both_layers = {k:v
                   for d in (anat_layers, func_layers)
                   for (k,v) in d.items()}
    layer_index = {k:ii for (ii,k) in enumerate(both_layers.keys())}
    saved_image_size = saved_image_size
    default_image_size = default_image_size
    def __init__(self, sids, features='func',
                 image_size=default_image_size,
                 cache_path=None):
        self.sids = np.array(sids)
        self.image_size = image_size
        if isinstance(features, tuple):
            for f in features:
                if f not in self.both_layers:
                    raise ValueError(f"unrecognized feature: {f}")
            self.features = features
        if features in ('func', 'anat', 'both'):
            self.features = features
        else:
            raise ValueError(f"features must be 'func', 'anat', or 'both'")
        self.cache_path = cache_path
        self._cache = {}
    @property
    def feature_count(self):
        """Returns the number of features (channels) in the dataset's images.

        If the dataset's `features` attribute is `'func'` or `'anat'` then this
        value is 4; if `features` is `'both'`, then `feature_count` is 8.
        """
        if isinstance(self.features, tuple):
            return len(self.features)
        else:
            return 8 if self.features == 'both' else 4
    @property
    def class_count(self):
        """Returns the number of classes (output labels) in the dataset.

        This is always 6.
        """
        return 6
    def fwd_transform(self, inimg, outimg=None):
        """Transforms an image in preparation for model training.

        Applies pre-model-training transformations to the input and output
        images and returns the resulting images. If one of the images is
        passed as `None`, then only the other image is transformed and returned.
        The transformations involve conveting the images into PyTorch tensors,
        ensuring they have the appropriate dtype, and putting the left and right
        hemisphere labels into separate channels.
        """
        if inimg is None and outimg is None: return None
        # The output image only needs to be to-tensor'ed
        if outimg is not None:
            tmp = np.asarray(outimg)
            hcols = tmp.shape[1] // 2
            outimg = np.zeros(tmp.shape[:2] + (6,), dtype=tmp.dtype)
            outimg[:,:hcols,:3] = tmp[:,:hcols]
            outimg[:,hcols:,3:] = tmp[:,hcols:]
            outimg = transforms.functional.to_tensor(outimg)
        # The input image needs to be normalized and to-tensor'ed
        if inimg is not None:
            inimg = transforms.functional.to_tensor(inimg)
        return (outimg if inimg  is None else
                inimg  if outimg is None else (inimg, outimg))
    def inv_transform(self, inimg, outimg=None):
        """Reverses the `fwd_transfom` transformation.

        Applies pre-model-training inverse transformations to the input and
        output images, either of which may be None (see also `fwd_transform`).
        """
        if inimg is None and outimg is None: return None
        if outimg is not None:
            outimg = np.transpose(outimg.numpy(), (1,2,0))
            outimg = outimg[:,:,:3] + outimg[:,:,3:]
        if inimg is not None:
            inimg = [np.clip(sl, 0, 1) for sl in inimg.numpy()]
            inimg = np.transpose(inimg, (1,2,0)).astype('float')
        return (outimg if inimg  is None else
                inimg  if outimg is None else (inimg, outimg))
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        (p,f,b,s) = self.images(self.sids[k],
                                image_size=self.image_size, 
                                cache=self._cache,
                                cache_path=self.cache_path)
        if isinstance(self.features, tuple):
            f = np.concatenate([p,f], axis=-1)
            p = f[..., [self.layer_index[k] for k in self.features]]
        else:
            p = (f if self.features == 'func' else
                 p if self.features == 'anat' else
                 b)
        return self.fwd_transform(p, s)
    def get(self, k):
        return self.images(self.sids[k],
                           image_size=self.image_size, 
                           cache=self._cache,
                           cache_path=self.cache_path)
    @staticmethod
    def resize_image(im, image_size):
        """Resizes the given image to the given size.

        Returns the image resized to have a number of rows equal to
        `image_size`. Uses scikit-imagee's `pyramid_expand` and `pyramid_reduce`
        to enlarge and reeduce imagees.
        """
        from skimage.transform import pyramid_expand, pyramid_reduce
        im = np.asarray(im)
        imsz = im.shape[0]
        if imsz < image_size:
            im = pyramid_expand(im, image_size/imsz, multichannel=True)
        elif imsz > image_size:
            im = pyramid_reduce(im, imsz/image_size, multichannel=True)
        return im
    @classmethod
    def images(self, sid, image_size=default_image_size, 
               cache={}, cache_path=None):
        """Returns the input and label images for the given subject-ID.

        Returns a tuple `(input_image, label_image)` for the given subject-ID
        `sid`; if possible, these images are loaded from the given `cache`
        object or `cache_path` directory.
        """
        im = cache.get(sid, None)
        if im is not None: return im
        found = False
        if cache_path is not None:
            iflnm = os.path.join(cache_path, 'images', '%s_anat.png' % sid)
            fflnm = os.path.join(cache_path, 'images', '%s_func.png' % sid)
            oflnm = os.path.join(cache_path, 'images', '%s_v123.png' % sid)
            try:
                with PIL.Image.open(iflnm) as f: im = np.array(f)
                param = im
                with PIL.Image.open(fflnm) as f: im = np.array(f)
                fparam = im
                with PIL.Image.open(oflnm) as f: im = np.array(f)
                sol = im
                cache[sid] = (param, fparam, sol)
                found = True
            except Exception: pass
        # If we haven't found the images in cache, generate them now.
        if not found: (param,fparam,sol) = self.generate_images(sid)
        # Resize the images if need-be.
        (param,fparam,sol) = [self.resize_image(im, image_size)
                              for im in (param, fparam, sol)]
        # Make a concatenation of anatomy and functional layers.
        bparam = np.concatenate([param, fparam], axis=-1)
        # Put them in the cache and save them if possible
        cache[sid] = (param, fparam, bparam, sol)
        if cache_path is not None and not found:
            flnm = os.path.join(cache_path, 'images', '%s_anat.png' % sid)
            im = np.clip(param, 0, 255).astype('uint8')
            PIL.Image.fromarray(im).save(flnm)
            flnm = os.path.join(cache_path, 'images', '%s_func.png' % sid)
            im = np.clip(fparam, 0, 255).astype('uint8')
            PIL.Image.fromarray(im).save(flnm)
            flnm = os.path.join(cache_path, 'images', '%s_v123.png' % sid)
            im = np.clip(sol, 0, 255).astype('uint8')
            PIL.Image.fromarray(im).save(flnm)
        return (param, fparam, bparam, sol)
    @classmethod
    def generate_images(self, sid, image_size=saved_image_size,
                        anat_layers=Ellipsis, func_layers=Ellipsis):
        """Generates and returns images for a single subject.

        Given a subject-id, generates and returns the tuple `(param_image,
        sol_iamge)` where `param_image` is the input image for the neural net
        and the `sol_image` is the output / solution to which the net should be
        trained.
        """
        dis = default_image_size
        if anat_layers is Ellipsis: anat_layers = HCPVisualDataset.anat_layers
        if func_layers is Ellipsis: func_layers = HCPVisualDataset.func_layers
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        # Get the subject and make a figure.
        sub = ny.data['hcp_lines'].subjects[sid]
        ms  = {h:ny.to_flatmap('occipital_pole', sub.hemis[h])
               for h in ['lh','rh']}
        ims = []
        for (p,(mn,mx)) in anat_layers.items():
            (fig,axs) = plt.subplots(1,2, figsize=(2,1), dpi=dis)
            fig.subplots_adjust(0,0,1,1,0,0)
            fig.set_facecolor('k')
            for (h,ax) in zip(['lh','rh'], axs):
                ax.axis('off')
                ax.set_facecolor('k')
                ny.cortex_plot(ms[h], color=p, axes=ax,
                               cmap='gray', vmin=mn, vmax=mx)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            bufstr = canvas.tostring_rgb()
            plt.close(fig)
            image = np.frombuffer(bufstr, dtype='uint8')
            image = np.reshape(image, (dis, dis*2, 3))
            ims.append(image[:,:,0])
        param = np.transpose(ims, (1,2,0))
        # Repeat for the functional param image
        ims = []
        for (p,(mn,mx)) in func_layers.items():
            (fig,axs) = plt.subplots(1,2, figsize=(2,1), dpi=dis)
            fig.subplots_adjust(0,0,1,1,0,0)
            fig.set_facecolor('k')
            for (h,ax) in zip(['lh','rh'], axs):
                ax.axis('off')
                ax.set_facecolor('k')
                # There is one special case: polar angle; we want to give the
                # rh and lh a similar map that is as non-circular as possible.
                if p.endswith('polar_angle'):
                    pp = ms[h].prop(p)
                    pp = -pp if h == 'rh' else pp
                    pp = np.mod(90 - pp + 180, 360) - 180
                else: pp = p
                ny.cortex_plot(ms[h], color=pp, axes=ax,
                               cmap='gray', vmin=mn, vmax=mx)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            bufstr = canvas.tostring_rgb()
            plt.close(fig)
            image = np.frombuffer(bufstr, dtype='uint8')
            image = np.reshape(image, (dis, dis*2, 3))
            ims.append(image[:,:,0])
        fparam = np.transpose(ims, (1,2,0))
        # Repeat for the solution image
        ims = []
        for lbl in [1,2,3]:
            (fig,axs) = plt.subplots(1,2, figsize=(2,1), dpi=dis)
            fig.subplots_adjust(0,0,1,1,0,0)
            fig.set_facecolor('k')
            for (h,ax) in zip(['lh','rh'], axs):
                ax.axis('off')
                ax.set_facecolor('k')
                ny.cortex_plot(ms[h], color=(ms[h].prop('visual_area') == lbl),
                               axes=ax, cmap='gray', vmin=0, vmax=1)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            bufstr = canvas.tostring_rgb()
            plt.close(fig)
            image = np.frombuffer(bufstr, dtype='uint8')
            image = np.reshape(image, (dis, dis*2, 3))
            ims.append(image[:,:,0])
        sol = np.transpose(ims, (1,2,0))
        return (param, fparam, sol)
def make_datasets(features=None,
                  sids=sids,
                  partition=default_partition,
                  cache_path=None,
                  image_size=default_image_size):
    """Returns a mapping of training and validation datasets.

    The mapping returned by `make_datasets()` contains, at the top level, the
    keys `'trn'` and `'val'` whose keys are the training and validation
    datasets, respectively. At the next level, the keys are `'anat'`, `'func'`,
    and `'both'` for the dataset input image type. The second level of maps are
    lazy.

    Parameters
    ----------
    features : 'func' or 'anat' or 'both' or None
        The type of input images that the dataset uses: functional data
        (`'func'`), anatomical data (`'anat'`), or both (`'both'`). If `None`
        (the default), then a mapping is returned with each input dataset type
        as values and with `'func'`, `'anat'`, and `'both'` as keys.
    sids : list-like, optional
        An iterable of subject-IDs to be included in the datasets. By default,
        the subject list `visual_autolabel.util.sids` is used.
    partition : partition-like
        How to make the partition of sujbect-IDs; the partition is made using
        `visual_autolabel.utils.partitoin(sids, how=partition)`.
    image_size : int, optional
        The width of the training images, in pixels (default: 512).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).

    Returns
    -------
    nested mapping of HCPVisualDataset objects
        A nested dictionary structure whose values at the bottom are datasets
        for training and validation partitions and for anatomy, function, and
        both. If `features` is `None`, then the return value is equivalent to
        `{f: make_datasets(f) for f in ['anat', 'func', 'both']}`.
    """
    (trn_sids, val_sids) = partition_sids(sids, how=partition)
    def curry_fn(sids, feat):
        return (lambda:HCPVisualDataset(sids, features=feat,
                                        cache_path=cache_path,
                                        image_size=image_size))
    if features is None:
        return pyr.pmap(
            {feat: pimms.lmap({'trn': curry_fn(trn_sids, feat),
                               'val': curry_fn(val_sids, feat)})
             for feat in ['anat', 'func', 'both']})
    else:
        return pimms.lmap({'trn': curry_fn(trn_sids, features),
                           'val': curry_fn(val_sids, features)})
def make_dataloaders(features=None,
                     sids=sids,
                     partition=None,
                     cache_path=None,
                     image_size=None,
                     datasets=None, 
                     shuffle=True,
                     batch_size=5):
    """Returns a pair of PyTorch dataloaders as a dictionary.

    `make_dataloaders('func')` returns training and validation dataloaders (in a
    dictionary whose keys are `'trn'` and `'val'`) for the functional data of
    HCP. The dataloaders and datasets can be modified with the optional
    arguments.

    Parameters
    ----------
    features : 'func' or 'anat' or 'both' or None, optional
        The type of input images that the dataset uses: functional data
        (`'func'`), anatomical data (`'anat'`), or both (`'both'`). If `None`
        (the default), then a mapping is returned with each input dataset type
        as values and with `'func'`, `'anat'`, and `'both'` as keys.
    sids : list-like, optional
        An iterable of subject-IDs to be included in the datasets. By default,
        the subject list `visual_autolabel.util.sids` is used.
    partition : partition-like, optional
        How to make the partition of sujbect-IDs; the partition is made using
        `visual_autolabel.utils.partitoin(sids, how=partition)`.
    image_size : int or None, optional
        The width of the training images, in pixels; if `None`, then 512 is
        used (default: `None`).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).
    datasets : None or mapping of datasets, optional
        A mapping of datasets that should be used. If the keys of this mapping
        are `'trn'` and `'val'` then all of the above arguments are ignored and
        these datasets are used for the dataloaders. Otherwise, if `features` is
        a key in `datasets`, then `datasets[features]` is used and the other
        options above are ignored. Otherwise, if `datasets` is `None` (the
        default), then the datasets are created using the above options.
    shuffle : boolean, optional
        Whether to shuffle the IDs when loading samples (default: `True`).
    batch_size : int, optional
        The batch size for samples from the dataloader (default: 5).

    Returns
    -------
    nested mapping of PyTorch DataLoader objects
        A nested dictionary structure whose values at the bottom are PyTorch
        data-loader objects for training and validation partitions and for
        anatomy, function, and both. If `features` is `None`, then the return
        value is equivalent to
        `{f: make_dataloader(f, **kw) for f in ['anat', 'func', 'both']}`.
    """
    if image_size is None: image_size = default_image_size
    # What were we given for datasets?
    if datasets is None:
        # We need to make the datasets using the other options.
        datasets = make_datasets(features=features, partition=partition,
                                 image_size=image_size, cache_path=cache_path)
    elif not pimms.is_map(datasets):
        raise ValueError("datasets must be a mapping or None")
    # If features is None, then we may need to produce a mapping of mappings of
    # datasets; otherwise, we dig in on the features key.
    if features is None:
        if not is_partition(datasets):
            def curry_fn(dset):
                return (lambda:make_dataloaders(datasets=dset,
                                                cache_path=cache_path,
                                                image_size=image_size,
                                                shuffle=shuffle,
                                                batch_size=batch_size))
            return pimms.lmap({k: curry_fn(v) for (k,v) in datasets.items()})
    elif features in datasets:
        datasets = datasets[features]
    # At this point, datasets must have 'trn' and 'val' entries in order to be
    # valid, or it must be a 2-tuple.
    if not is_partition(datasets):
        raise ValueError("make_dataloaders(): provided datasets are not valid")
    # Okay, now we can make the data-loaders using these datasets.
    trn = trndata(datasets)
    val = valdata(datasets)
    return pyr.m(
        trn=DataLoader(trn, batch_size=batch_size, shuffle=shuffle),
        val=DataLoader(val, batch_size=batch_size, shuffle=shuffle))

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
        if self.apply_sigmoid: out = torch.sigmoid(out)
        return out
