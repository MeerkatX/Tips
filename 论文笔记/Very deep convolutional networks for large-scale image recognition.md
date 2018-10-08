# VGG Net

解决ImageNet中的1000类图像分类和定位问题，VGG16和VGG19表现最好

#### 特点

- 小卷积核。作者将卷积核全部替换为3x3（极少用了1x1）；

- 小池化核。相比AlexNet的3x3的池化核，VGG全部为2x2的池化核；
- 层数更深特征图更宽。基于前两点外，由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，计算量的增加放缓；
- 全连接转卷积。网络测试阶段将训练阶段的三个全连接替换为三个卷积，测试重用训练时的参数，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高的输入。

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

- 结论：

  1. 同样stride下，不同卷积核大小的特征图和卷积参数差别不大

  2. 越大的卷积核计算量越大

## Training

#### 参数设置
- using **mini-batch gradient descent with momentum**.The batch size was set to **256**,momentum to **0.9**. 随机（批量）梯度下降+动量

- L2正则以及dropout. The training was regularised by weight decay (the L2 penalty multiplier set to $5*{10^{-4}}$) and dropout regularisation for the first two fully-connected layers (dropout ratio set to **0.5**).

- 学习率设置，当准确度不再提升时减少学习率 The learning rate was initially set to $10^{−2}​$. decreased by a factor of 10 when the validation set accuracy stopped improving
#### 参数weights的初始化
- 先训练一个浅的模型 began with training the configuration A, shallow enough to be trained with random initialisation

- 当训练更深的模型的时候 we initialised the first four convolutional layers and the last three fullyconnected layers with the layers of net A (the intermediate layers were initialised randomly).中间层随机初始化参数，from a **normal distribution** with zero mean and $10^{-2}$ variance。**biases** = 0

- 并且在这个训练中不减少学习率 We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning.

- 从rescaled training images随机裁切 每次梯度下降迭代每张图片裁切一次(one crop per image per SGD iteration)这样可以充实训练数据，分别有random horizontal flipping and random RGB colour shift(随机水平移动和随机RGB色移)

- 有两种训练数据缩放方式 (VGG出于计算速度的考虑使用了single-scale)

  1. single-scale

     isotropically-rescaled 各项同性缩放，裁剪为224*224，固定的尺寸

  2. multi-scale 

     each training image is individually rescaled by randomly sampling S from a certain range [Smin,Smax] (we used Smin = 256 and Smax = 512).

#### Testing

- 全连接转卷积 可以接收任意宽或高的输入(测试，训练的宽高不同) **将卷积核大小设置为输入的空间大小.**

## classification experiments

#### single-scale evaluation

#### muti-scale evaluation


## 其它

#### TensorFlow版github VGG16,19代码

[代码](https://github.com/machrisaa/tensorflow-vgg)

#### LRN局部响应归一化

- 侧抑制(lateral inhibition)被激活神经元抑制相邻神经元

- 有利于增加泛化能力，公式如下：
  $$
  b^i_{x,y}=a^i_{x,y}/\left(k+\alpha\sum_{j=\max(0,i-\frac{n}{2})}^{\min(N-1,i+\frac{n}{2})}(a_{x,y}^j)^2\right)^\beta
  其中一般k=2,\alpha=1^{10^{-4}},n=5,\beta=0.75
  $$




#### 各项同性缩放和各向异性缩放

- 各向异性缩放，不管图片比例，扭曲，直接缩放
- 各项同性缩放
    - 先扩充后裁剪

      直接在原始图片中，把bounding box的边界进行扩展延伸成正方形，然后再进行裁剪；如果已经延伸到了原始图片的外边界，那么就用bounding box中的颜色均值填充

    - 先裁剪后扩充

      先把bounding box图片裁剪出来，然后用固定的背景颜色填充成正方形图片(背景颜色也是采用bounding box的像素颜色均值)
## reference

[深度学习VGG模型核心拆解](https://blog.csdn.net/qq_40027052/article/details/79015827)

