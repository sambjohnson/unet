# unet
UNet-based cortical parcellation

This repository contains experiments using a U-Net CNN for cortical parcellation. There are a few key features to the approach.

- Architecture: relatively simple. A Resnet18 downswing and straightforward upsampling + Conv layers on the upswing.  
- Training: transfer learning approach; pretraining on a large dataset of automatically generated (Freesurfer -- Destrieux) parcellations.  
- Augmentation: angle jitter (in underlying 3D space) and translation.  

This code is adapted (and in places, substantially pruned) from Noah Benson's project for labeling the visual cortex, V1 - V4.
