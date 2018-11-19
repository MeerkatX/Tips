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

## Reference

梯度消失/爆炸：

因为bp传递梯度，层数很多时，如果梯度<1的话，多个<1的梯度连乘最后会得到一个很小的值，即梯度消失。导致前面的层参数无法更新

[为什么ResNet和DenseNet可以这么深](https://zhuanlan.zhihu.com/p/28124810?group_id=883267168542789632)

[残差网络ResNet笔记](https://www.jianshu.com/p/e58437f39f65)

