# Faster R-CNN

## abstract:

为了解决区域提议的计算瓶颈，提出了**RPN** Region Proposal Network (RPN) that **shares full-image convolutional features with the detection network** （与检测网络(fast R-CNN)共享整个图像的卷积特征）

An RPN is a fully convolutional network that simultaneously **predicts object bounds and objectness scores** at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. 

将RPN和Fast RCNN结合成一个网络(通过卷积特征的共享)，并且使用了**注意力**机制，来使得RPN网络告诉整个网络应该看哪个位置。

*将区域提议和检测网络合二为一，原先的区域提议算法都是基于CPU的，计算慢，为什么不将其改为GPU实现，原文有相关原因。并且与之前R-CNN 2000个区域建议不同的是只提供了300个建议，以及提出了anchor box，在之后的SDD中也有相关应用*

## Reference

[Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)