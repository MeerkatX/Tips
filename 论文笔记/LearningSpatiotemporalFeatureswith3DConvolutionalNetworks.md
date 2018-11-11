# Learning Spatiotemporal Features with 3D Convolutional Networks

## 简介

2D convolution 能获得很好的空间信息，但不能很好的捕获时序上信息，3D convolution 可以获取时序信息，可以作为一个通用网络，之后的 3D U-net会用到。

## 3D卷积：

### 卷积示意图

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/c3d.png)

for our architecture search study we ﬁx the spatial receptive ﬁeld to $3×3$ and vary only the temporal depth of the 3D convolution kernels. 

from now on we refer video clips with a size of $c×l×h×w$ where $c$ is the number of channels, $l$ is length in number of frames, $h$ and $w$ are the height and width of the frame

We also refer 3Dconvolutionandpoolingkernelsizeby $d×k×k$,where d is kernel temporal depth and k is kernel spatial size.

基于3D卷积操作，作者设计了如上图所示的C3D network结构。共有8次卷积操作，4次池化操作。其中卷积核的大小均为$3\times3\times3$，步长为 $1\times1\times1$。池化核的大小为$2\times2\times2$,步长为$2\times 2\times2$，但第一层池化除外，其大小和步长均为$1\times 2\times 2$。这是为了不过早缩减时序上的长度。最终网络在经过两次全连接层和softmax层后就得到了最终的输出结果。网络的输入尺寸为$3\times16\times112\times 112$，即一次输入16帧图像。

### Common network settings: 

All video frames are resized into 128×171. **训练阶段随机裁剪** We also use jittering by using random crops with a size of $3×16×112×112$ of the input clips during training. 

 Videos are split into non-overlapped 16-frame clips which are then used as input to the networks.**输入维度**一次输入16帧图片： The input dimensions are $3×16×128×171$.

5个卷积层，5个池化层（each convolution layer is immediately followed by a pooling layer），2个全连接层和softmax loss层用来预测。 卷积核个数分别为64 128 256 256 256。

#### 卷积核

All convolution kernels have a sizeof d where d is the kernel temporal depth 经过测试 $d=3$ 比较好.卷积都是padding same，stride 1这样。   **3x3x3**

#### 池化层

All pooling layers are max pooling with kernel size $2×2×2$ (except for the ﬁrst layer) with stride 1 which means the size of output signal is reduced by a factor of 8 compared with the input signal. 除了第一层。

第一层是$ 1 × 2 × 2 $ 因为不想过早这是为了不过早缩减时序上的长度。原文叙述如下：

（not to merge the temporal signal too early and also to satisfy the clip length of 16 frames (e.g. we can temporally pool with factor 2 at most 4 times before completely collapsing the temporal signal). ）

#### 训练

We train the networks from scratch using mini-batches of  30 clips,with initial learning rate of 0.003. The learning rate is divided by 10 after every 4 epochs. The training is stopped after 16 epochs. 

Training is done by SGD with minibatch size of 30 examples. 也就是说 训练阶段，一次训练了30 mini-batches 每个例子有 3 chanels 16 frames h w 四个参数

## 结论：

C3D使用3D CNN构造了一个效果不错的网络结构，对于基于视频的问题均可以用来提取特征。可以将其全连接层去掉，将前面的卷积层放入自己的模型中，就像使用预训练好的VGG模型一样。这里主要是作为之后可能会用到的3D U-Net的预先知识。

## Reference：

