![architecture](models/nvidia_model-achitecture.png)

NVIDIA model used

Image normalization to avoid saturation and make gradients work better.

Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU

Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU

Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU

Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU

Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU

Drop out (0.5)

Fully connected: neurons: 100, activation: ELU
Fully connected: neurons: 50, activation: ELU
Fully connected: neurons: 10, activation: ELU
Fully connected: neurons: 1 (output)