# Fully Convolutional Networks for Semantic Segmentation(语义分割)

## Abstract

提出了FCN全卷积网络来做Dense Prediction像素级预测，即不在意输入图片大小

Our key insight is to build “fully convolutional” networks that take input of **arbitrary size** and produce **correspondingly-sized** output with efﬁcient inference and learning. 

## Fully  convolutional networks



## 反卷积/转置卷积

![img](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/no_padding_no_strides_transposed.gif)

这里说明了反卷积的时候，是有补0的，即使人家管这叫no padding（$p=0$），这是因为卷积的时候从蓝色 4×4  缩小为绿色 $2×2$，所以对应的 $p=0$ 反卷积应该从蓝色 $2×2$ 扩展成绿色 $4×4$。而且转置并不是指这个 $3×3$  的核 $w$  变为 $w^T$ ，但如果将卷积计算写成矩阵乘法（在程序中，为了提高卷积操作的效率，就可以这么干，比如tensorflow中就是这种实现）， $\overrightarrow{Y}=C\overrightarrow{X}$（其中 $ \overrightarrow{Y}$ 表示将 $\overrightarrow{Y}$ 拉成一维向量， $\overrightarrow{X}$ 同理），那么反卷积确实可以表示为 $C^T\overrightarrow{Y}$，而这样的矩阵乘法，恰恰等于 $w$ **左右翻转再上下翻转**(旋转180°)后与补0的 $Y​$ 卷积的情况。

## Reference

[深度学习 | 反卷积/转置卷积 的理解 transposed conv/deconv](https://blog.csdn.net/u014722627/article/details/60574260)

#### Patch-wise training

An early approach to segmentation was to run patches of pixels, centered around the pixel of interest, through a CNN classifier ([Ciresan et al. 2012](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf)). Doing this for each pixel produces the segmented image. Following[Akkus et al.](https://link.zhihu.com/?target=https%3A//www.ncbi.nlm.nih.gov/pmc/articles/PMC5537095/)(2017), we will call this “patch-wise” segmentation.

未完待续…