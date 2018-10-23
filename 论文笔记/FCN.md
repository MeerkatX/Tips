# Fully Convolutional Networks for Semantic Segmentation(语义分割)

## Abstract

提出了FCN全卷积网络来做Dense Prediction像素级预测，即不在意输入图片大小

Our key insight is to build “fully convolutional” networks that take input of **arbitrary size** and produce **correspondingly-sized** output with efﬁcient inference and learning. 

## Fully  convolutional networks



## Reference



#### Patch-wise training

An early approach to segmentation was to run patches of pixels, centered around the pixel of interest, through a CNN classifier ([Ciresan et al. 2012](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf)). Doing this for each pixel produces the segmented image. Following[Akkus et al.](https://link.zhihu.com/?target=https%3A//www.ncbi.nlm.nih.gov/pmc/articles/PMC5537095/)(2017), we will call this “patch-wise” segmentation.