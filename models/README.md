# Models

## model_trainer.py
Modularized Trainingspipeline for consistency

## resnet_lidar
provides Architectures and weights for the lidar model

## resnet_rgb
provides Architectures and weights for the rgb model. Also contains development notebooks for experimentation without messing around with the model_trainer which is meant to be consistent.

### Architectures
All architecture are inspired by the following Papers:
- End-to-end Driving via Conditional Imitation Learning https://arxiv.org/abs/1710.02410
- Exploring the Limitations of Behavior Cloning for Autonomous Driving Exploring the Limitations of Behavior Cloning for Autonomous Driving  https://arxiv.org/abs/1904.08980
- Multimodal E2E AD https://arxiv.org/abs/1906.03199
