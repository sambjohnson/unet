# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/config.py
# Configuration of the visual_autolabel package.
# This file just contains global definitions that are used throughout.

"""Global configuration variables for the `visual_autolabel` package.

The `visual_autolabel.config` package contains definitions of global variables
that are used throughout the package.

Attributes
----------
saved_image_size : int
    The size of images (number of image rows) that get saved to cache. It's
    relatively efficient to downsample images, so keeping this somewhat larger
    than needed is a good idea. The value used in the package is 512.
default_image_size : int
    The default size for images used in model training. This is the number of
    rows in the images.
default_partition : tuple
    The default way of partitioning the subjects into training and validation
    datasets. See also the `visual_autolabel.partition` function. The value used
    in the package is `(0.8, 0.2)`, indicating that 80% of subjects should be
    in the training dataset, and 20% of the subjects should be in the valudation
    dataset.
sids : NumPy array of ints
    The subject IDs of the HCP subjects used in the training datasets
"""


#-------------------------------------------------------------------------------
# saved_image_size
# The size of images (number of image rows) that get saved to cache. It's
# relatively efficient to downsample images, so keeping this somewhat larger
# than needed is a good idea.
saved_image_size = 512

#-------------------------------------------------------------------------------
# default_image_size
# The default size for images used in model training. This is the number of rows
# in the images.
default_image_size = 128

#-------------------------------------------------------------------------------
# default_partition
# The default training-validation partition. This should be a tuple of either
# the subject count for each category or the subject fraction for each category.
# The training fraction is listed first, and the validation fraction is listed
# second.
default_partition = (0.8, 0.2)

#-------------------------------------------------------------------------------
# sids
# The subject-IDs that we use in training.
# We need a couple libraries to generate the list, but we delete them below.
# import neuropythy as ny, numpy as np
# sids = np.array([sid for sid in ny.data['hcp_lines'].subject_list
#                 if ('mean',sid,'lh') not in ny.data['hcp_lines'].exclusions
#                  if ('mean',sid,'rh') not in ny.data['hcp_lines'].exclusions])
# sids.flags['WRITEABLE'] = False
# We can delete these from the module now that we're done with them.
# del ny, np


#===============================================================================
# __all__
__all__ = ["saved_image_size",
           "default_image_size",
           "default_partition"
#           ,"sids"
           ]
