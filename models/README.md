# Models

## model_trainer.py
Modularized trainings pipeline for consistency.

## resnet_lidar
Provides architectures and weights for the lidar model.
<!-- ![System architecture of project](../figures/lidar_model.png "LiDAR model") -->
<p align="center">
  <img src="../figures/lidar_model.png" width="350" title="hover text">
</p>

## resnet_rgb
Provides architectures and weights for the rgb model. Also contains development notebooks for experimentation without messing around with the model_trainer which is meant to be a stable version.
<p align="center">
  <img src="../figures/camera_model.png" width="350" title="hover text">
</p>

### Architectures
All architecture are inspired by the following Papers:
- End-to-end Driving via Conditional Imitation Learning https://arxiv.org/abs/1710.02410
- Exploring the Limitations of Behavior Cloning for Autonomous Driving Exploring the Limitations of Behavior Cloning for Autonomous Driving  https://arxiv.org/abs/1904.08980
- Multimodal E2E AD https://arxiv.org/abs/1906.03199
