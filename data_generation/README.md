# Data Generation

## TransFuser Dataset

This project primarily uses the Transfuser (https://github.com/autonomousvision/transfuser) Dataset to train models. Dataset contains rgb, lidar, depth, semantic segmentation and topdown(birds eye view) sensors as well as metadata, however we have restricted our models to train on two presets of data types (rgb only and rgb + lidar). The dataset can be installed either via (taken from TransFuser Project)

```Shell
chmod +x download_data.sh
./download_data.sh
```

or can be manually created with the Carla Client and the scripts provided in the TransFuser repository.

## Noise Injection

We have used noise injected data in our training to improve our agents. Since the existing dataset comes without any noise, we have used the TransFuser data generation script and a Carla Client running in a local computer to create our own noisy data. Noise is injected in intervals of 10 seconds (Cycles of 20 seconds, 10 seconds of noise followed by 10 second of recovery and data generation during recovery). The aforementioned script is provided in the data_generation folder. 