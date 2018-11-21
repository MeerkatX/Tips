# Deep Residual Learning for Image Recognition

之前看吴恩达的deeplearning.ai课中有相关残差网络的介绍，只是大致了解了一点残差网络。之后发现很多地方都要用到残差，所以又认真看了一下论文。

## Abstract

深度神经网络 VGG16 或 19 ，googLeNet等深度不断加深，所以 Is learning better networks as easy as stacking more layers? 

问题是否定的，其中会有很多问题，比如梯度消失/爆炸。越深的网络越难去训练，所以提出了残差学习框架(residual learning framework) 去解决这个训练问题  即原先是拟合 $H(x)$ 改变之后是 $F(x):=H(x)+x$ 即加上了层输入 $x$  ( We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. )

结果是 these residual networks are easier to optimize,and can gain accuracy from considerably increased depth. 

## Introduction

#### 为什么提出残差网络？

如之前提到的**梯度消失/爆炸**，这个问题在很大程度上已经被**batch normalization**解决了，但是还有个问题：

**退化问题(degradation)**：网络层数增加，但是在训练集上的准确率却饱和甚至下降了。这个不能解释为overfitting，它应该表现为在训练集上表现更好才对。退化问题说明了深度网络不能很简单地被很好地优化。就如图这样：

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/resnet.png)

通过浅层网络加上 y=x 等同映射构造深层模型，结果深层模型并没有比浅层网络有等同或更低的错误率，推断退化问题可能是因为深层的网络并不是那么好训练，也就是**求解器很难去利用多层网络拟合同等函数**。

#### 如何解决?

如果深层网络的后面那些层是**恒等映射**，那么模型就退化为一个浅层网络。那现在要解决的就是学习恒等映射函数了。 但是直接让一些层去拟合一个潜在的恒等映射函数H(x) = x，比较困难，这可能就是深层网络难以训练的原因。但是，如果把网络设计为H(x) = F(x) + x,如下图。我们可以转换为学习一个残差函数F(x) = H(x) - x. 只要F(x)=0，就构成了一个恒等映射H(x) = x. 而且，拟合残差肯定更加容易。

#### 其他解释

- 差分放大器

## Deep Residual Learning

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/resnet2.png)

#### Residual Learning

在残差学习重构的情况下, 如果特征映射是最优的, 它可能只是将多个非线性层的权重推向零, 以接近特征映射。

With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

如果dim相同的话（加是逐个元素相加）：
$$
y=F(x,\{W_i\})+x
$$
如果dim不同的话：
$$
y=F(x,\{W_i\})+W_sx
$$
即一个$1\times1$的卷积修改x的通道数（thus $W_s$ is only used when matching dimensions. ），以及如果降采样的话，利用的是strades设置为2来降采样，如果特征图不同的话，用$1\times 1$卷积加stride=2 （ For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2. ）

**用卷积层进行残差学习：**以上的公式表示为了简化，都是基于全连接层的，实际上当然可以用于卷积层。加法随之变为对应channel间的两个feature map逐元素相加。

更深的网络结构中，可以用右边，降维减少计算量，计算后再增加维度：

![img](https://upload-images.jianshu.io/upload_images/6095626-287fc59a3cd86488.png?imageMogr2/auto-orient/)

## 网络结构：

![img](https://upload-images.jianshu.io/upload_images/6095626-2c3d2b4c683ec4ac.png?imageMogr2/auto-orient/)

设计网络的规则：1.对于输出feature map大小相同的层，有相同数量的filters，即channel数相同；2. 当feature map大小减半时（池化），filters数量翻倍。
对于残差网络，维度匹配的shortcut连接为实线，反之为虚线。维度不匹配时，同等映射有两种可选方案：

1. 直接通过zero padding 来增加维度（channel）。
2. 乘以W矩阵投影到新的空间。实现是用1x1卷积实现的，直接改变1x1卷积的filters数目。这种会增加参数。

## Reference

梯度消失/爆炸：

因为bp传递梯度，层数很多时，如果梯度<1的话，多个<1的梯度连乘最后会得到一个很小的值，即梯度消失。导致前面的层参数无法更新

[为什么ResNet和DenseNet可以这么深](https://zhuanlan.zhihu.com/p/28124810?group_id=883267168542789632)

[残差网络ResNet笔记](https://www.jianshu.com/p/e58437f39f65)

