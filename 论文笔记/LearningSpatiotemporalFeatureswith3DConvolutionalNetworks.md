# Learning Spatiotemporal Features with 3D Convolutional Networks

## 简介

2D convolution 能获得很好的空间信息，但不能很好的捕获时序上信息，3D convolution 可以获取时序信息，可以作为一个通用网络，之后的 3D U-net会用到。

## 3D卷积：

[img]()

for our architecture search study we ﬁx the spatial receptive ﬁeld to $3×3$ and vary only the temporal depth of the 3D convolution kernels. 

from now on we refer video clips with a size of $c×l×h×w$ where $c$ is the number of channels, $l$ is length in number of frames, $h$ and $w$ are the height and width of the frame

