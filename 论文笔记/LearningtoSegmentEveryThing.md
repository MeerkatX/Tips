# Learning to Segment Every Thing

## Abstract

因为做分割需要mask，而做数据标注的话很难。This requirement makes it expensive to annotate new categories and has restricted instance segmentation models to∼100 well-annotated classes. 

所以想着去做部分监督学习 partially supervised training paradigm，用一个weight  transfer function来做mask的预测。

training instance segmentation models on a large set of categories all of which have box annotations, but only a small fraction of which have mask annotations. 可以训练一个利用bounding box的分割模型，然后只需要比较少的已经标注了mask的模型。即只含有bb的和同时含有bb和mask的训练集B,A来进行训练。

模型是再mask RCNN的基础上改进的。

## Learning to Segment Every Thing

#### 结构图：

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/learningtos.png)

#### Mask Prediction Using Weight Transfer

因为之前还没看mask R-CNN，所以简介：

In brief, Mask R-CNN can be seen as augmenting a Faster R-CNN [34] bounding box detection model with an additional mask branch that is a small fully convolutional network (FCN) [24]. 



