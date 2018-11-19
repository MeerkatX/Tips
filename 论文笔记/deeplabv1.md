# SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS

google提出的一种语义图像分割的方式，deeplabv1 2015

## Abstract

深度卷积神经网络在高层的视觉任务表现很好，比如图像分类和目标检测。但是在像素级的任务表现有限 poor localization property

deeplabv1将DCNNs和概率图模型结合来解决像素级的分类问题（语义分割）：

combining the responses at the **ﬁnal DCNN layer** with a **fully connected Conditional Random Field** (CRF) 条件随机场

以及使用了带孔卷积`atrous`算法来扩大感受野，加速GPU计算

Careful network re-purposing and a novel application of the ’hole’ algorithm from the wavelet community allow dense computation of neural net responses at 8 frames per second on a modern GPU.

## Introduction

#### DCNNs为什么不利于解决像素级任务

There are two technical hurdles in the application of DCNNs to image labeling tasks: signal down-sampling, and spatial ‘insensitivity’ (invariance). **信号降采样** 和 **空间不变性** 

- 降采样主要是max-pooling以及卷积时striding操作

- 空间不变性 分类器获取以对象中心的决策是需要空间变换的不变性，这天然的限制了DCNN的定位精度

(obtaining object-centric decisions(获取以物体为中心的决策) from a classiﬁer requires invariance to spatial transformations, inherently limiting the spatial accuracy of the DCNN model.)

#### deeplabv1提出的两个解决方案

针对降采样，采用了`atrous`算法(with holes)带孔卷积 This allows efﬁcient dense computation of DCNN responses in a scheme substantially simpler than earlier solutions to this problem

针对空间不变性，采用了全连接的条件随机场( fully-connected Conditional Random Field (CRF).)

#### 当时的优点

- 速度(DCNN 8fps速度，CRF需要0.5s)，准确(语义分割比赛第二)，简单(DCNN+CRFs)

## Convolutional Neural Networks For Dense Image Labeling

首先就网络结构来说，首先将VGG16的最后几个全连接层变为全卷积层，7x7x4096等等，之后就如下：

#### 带孔卷积和more densely detection scores

在图像的原始分辨率上产生非常稀疏的计算检测分数(步幅32,步幅=输入尺寸/输出特征尺寸步幅)，为了以更密集(步幅8)的计算得分,我们在最后的两个最大池化层不下采样(padding到原大小)，再通过2或4的采样率的空洞卷积对特征图做采样扩大感受野，缩小步幅。

因为最后两个最大池化被跳过了，所以要做到和以前类似的感受野，就采用了带孔卷积。

这种带孔的采样又称**atrous**算法，可以稀疏的采样底层特征映射，该方法具有通用性，并且可以使用任何采样率计算密集的特征映射。在VGG16中使用不同采样率的空洞卷积，可以让模型再密集的计算时，明确控制网络的感受野。保证DCNN的预测图可靠的预测图像中物体的位置。训练时将预训练的VGG16的权重做fine-tune，损失函数取是输出的**特征图**与**ground truth下采样8倍**做**交叉熵和**；All positions and labels are equally weighted in the over all loss function. 

测试时取**输出图双线性上采样8倍**得到结果。*(主要是因为 类分数映射(对应于对数概率)非常平滑，这允许我们使用简单的双线性插值来增加它们的分辨率8倍，而计算成本可以忽略不计)*

但DCNN的预测物体的位置是粗略的，没有确切的轮廓。在卷积网络中，因为有多个最大池化层和下采样的重复组合层使得模型的具有平移不变性，我们在其输出的high-level的基础上做定位是比较难的。这需要做分类精度和定位精度之间是有一个自然的折中。

#### 控制接受域以及加速卷积网络的密集计算

Most recent DCNN-based image recognition methods rely on networks pre-trained on the Imagenet large-scale classiﬁcation task. These networks typically have large receptive ﬁeld size: in the case of the VGG-16 net we consider, its receptive ﬁeld is 224×224 (with zero-padding) and 404×404 pixels if the net is applied convolutionally. 

当转化为全卷积时，$7 \times 7 \times 4096$ 的空间大小变成计算score map的瓶颈，所以调整为 $4\times4 $ 的卷积核。

原文如下：

We have addressed this practical problem by spatially subsampling (by simple decimation) the ﬁrst FC layer to 4×4 (or 3×3) spatial size. This has reduced the receptive ﬁeld of the network down to 128×128(withzero-padding)or308×308(inconvolutionalmode)andhasreducedcomputationtime for the ﬁrst FC layer by 2−3 times. 

## 解决分类精度和定位精度权衡的问题

- 利用卷积网络中多个层次信息  information from multiple layers 
- 采样超像素表示，将定位任务交给低级(底层的特征图)的分割  super-pixel representation

在deeplabv1利用了fully connected CRFs来使得定位更准确。

#### Fully Connected Conditional Random Fields For Accurate Localization

传统上条件随机场是用来平滑分割图的噪声，但是这里已经很平滑了，需要的是找到准确边界。

通常，这些模型包含耦合相邻节点的能量项，有利于相同标签分配空间近端像素。定性的说，这些短程的CRF主要功能是清除在手工特征基础上建立的弱分类器的虚假预测。与这些弱分类器相比，现代的DCNN体系产生质量不同的预测图，通常是比较平滑且均匀的分类结果(即以前是弱分类器预测的结果，不是很靠谱，现在DCNN的预测结果靠谱多了)。在这种情况下，使用短程的CRF可能是不利的，因为我们的目标是恢复详细的局部结构，而不是进一步平滑。而有工作证明可用全连接的CRF来提升分割精度。

To over come these limitations of short-range CRFs,we integrate into our system the fully connected CRF model of `Kr¨ahenb¨uhl & Koltun (2011)`. The model employs the energy function ：
$$
E(x)=\sum_{i}\theta_i(x_i)+\sum_{ij}\theta_{ij}(x_i,x_j)
$$
x是分配到像素的label，一元势函数$\theta_i(x_i)=-\log P(x_i)$ 其中$P(x_i)$是分配到像素的label的概率，是由DCNN计算出来的。二元势函数$\theta_{ij}(x_i,x_j)=\mu(x_i,x_j)\sum^k_{m=1}w_m \cdot k^m(f_i,f_j) $  其中如果$x_i\neq x_j$ 的话，$\mu(x_i,x_j)=1$，其余等于0 ，其中一对像素 $i ,j$ 无论在图片距离多远位置都计算二元势函数。

#### 多尺度

之后尝试了多尺度，但是效果不如CRFs

## Reference

[读懂概率图模型](https://zhuanlan.zhihu.com/p/31527050)

[FCN->deeplab](https://blog.csdn.net/junparadox/article/details/52610744)

[deeplabv1+deeplabv2+带孔卷积详细的讲解](https://blog.csdn.net/ming0808sun/article/details/78843471)