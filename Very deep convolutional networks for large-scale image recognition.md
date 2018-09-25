# VGG

## ConvNet Configurations

#### 网络结构

- 输入的图片为fixed-size 224 * 224 RGB image，预处理为减去RGB的均值，以下为VGG tensorflow源码:

  ```python
  red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
  assert red.get_shape().as_list()[1:] == [224, 224, 1]
  assert green.get_shape().as_list()[1:] == [224, 224, 1]
  assert blue.get_shape().as_list()[1:] == [224, 224, 1]
  bgr = tf.concat(axis=3, values=[
       blue - VGG_MEAN[0],
       green - VGG_MEAN[1],
       red - VGG_MEAN[2],
  ])#VGG_MEAN已经算过，VGG_MEAN = [103.939, 116.779, 123.68]
  ```

- 使用了很小的感知野(receptive field) **3*3**  convolution filters，在之后的C结构(共有6种结构)还用了 **1*1** 的filter(可以被看作线性变换)，其中**stride**被固定为**1**个像素，**pading设置为** the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution(即添加padding保持原有的尺寸大小) 
  $$
  H_{output} = 1 + \lfloor{\frac{(H_{input} + 2 * pad - H_{filtersize})}{stride}}\rfloor
  $$
  padding是在每个max-pooling(共5个)用，**max-pooling**设置为**2*2** stride为**2**

- convolutional layer + fully-connected layers (fc first two have **4096** channels,third performs **1000** channels) + softmax layer

- All hidden layers之后跟着ReLU。LRN(局部响应归一化)只有一个包含(A-LRN)，论文中认为LRN并没有效果反而增加了存储空间和计算时间

## Training

#### 参数设置
- using **mini-batch gradient descent with momentum**.The batch size was set to **256**,momentum to **0.9**. 随机（批量）梯度下降+动量

- L2正则以及dropout. The training was regularised by weight decay (the L2 penalty multiplier set to $5^{10^{-4}}$) and dropout regularisation for the first two fully-connected layers (dropout ratio set to **0.5**).

- 学习率设置，当准确度不再提升时减少学习率 The learning rate was initially set to $10^{−2}$. decreased by a factor of 10 when the validation set accuracy stopped improving
#### 参数初始化
- 先训练一个浅的模型 began with training the configuration A, shallow enough to be trained with random initialisation

- 当训练更深的模型的时候 we initialised the first four convolutional layers and the last three fullyconnected layers with the layers of net A (the intermediate layers were initialised randomly).

- 并且在这个训练中不减少学习率 We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning.

#### TensorFlow版github VGG16,19代码

#### LRN局部响应归一化

- 侧抑制(lateral inhibition)被激活神经元抑制相邻神经元

- 有利于增加泛化能力，公式如下：
  $$
  b^i_{x,y}=a^i_{x,y}/\left(k+\alpha\sum_{j=\max(0,i-\frac{n}{2})}^{\min(N-1,i+\frac{n}{2})}(a_{x,y}^j)^2\right)^\beta
  其中一般k=2,\alpha=1^{10^{-4}},n=5,\beta=0.75
  $$

