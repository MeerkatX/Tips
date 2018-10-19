# Faster R-CNN

## abstract:

为了解决区域提议的计算瓶颈，提出了**RPN** Region Proposal Network (RPN) that **shares full-image convolutional features with the detection network** （与检测网络(fast R-CNN)共享整个图像的卷积特征）

An RPN is a fully convolutional network that simultaneously **predicts object bounds and objectness scores** at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. 

将RPN和Fast RCNN结合成一个网络(通过卷积特征的共享)，并且使用了**注意力**机制，来使得RPN网络告诉整个网络应该看哪个位置。

*将区域提议和检测网络合二为一，原先的区域提议算法都是基于CPU的，计算慢，为什么不将其改为GPU实现，原文有相关原因。并且与之前R-CNN 2000个区域建议不同的是只提供了300个建议，以及提出了anchor box，在之后的SDD中也有相关应用*

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnn2.png)

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnn1.jpg)

## Anchors

在最后一层的卷积层得到的特征图上使用滑动窗口，同时预测是否含有object以及回归anchors，结构示意如下：

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnn3.png)

#### 分类层

我们对每个锚点输出两个预测值：它是背景（不是目标）的分数，和它是前景（实际的目标）的分数

#### 回归或边框调整层

我们输出四个预测值：`x_center`、`y_center`、`width`、`height`，我们将会把这些值用到锚点中来得到最终的建议

#### anchors各有3种长宽比和比例，所以共有9种anchors：

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnnanchors.jpg)

数学上，如果图片的尺寸是 $w\times h$，那么特征图最终会缩小到尺寸为 $\frac{w}{r}$ 和 $\frac{h}{r}$，其中 r 是次级采样率。如果我们在特征图上每个空间位置上都定义一个锚点，那么最终图片的**锚点会相隔 r 个像素**，在 VGG16 中，r=16 大概就是这个样子：

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnn4.jpg)

如果卷积后的特征图为 $W\times H$ 的话，就总共有 $WHk$ 个anchors。

#### 平移不变性`Translation-Invariant` 

即如果调整图片将目标平移了，应该仍然能够检测出这个目标。

其中参数计算是（VGG16为例）：

滑动窗口，vgg16最后一层卷积层的特征图上，利用$3\times3\times512$的卷积滑动，共512个卷积核，每次滑动生成512个参数，这512个参数进行anchor回归，分类 *2k* `cls layer` *4k* `reg layer`

所以最后计算参数量是 $3\times3\times512\times512+512\times(4+2)\times9$ 

## loss Function

首先是**正例** positive label：

1. the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box
2.  an anchor that has an IoU overlap higher than 0.7 with any ground-truth box

正例中会有`ground-truth box`匹配多个anchor，都看作正例。那么只有条件2应该就满足要求了，论文中解释如下：

 in some rare cases the second condition may ﬁnd no positive sample.

之后是**反例** negative label:

- negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. 

既不是正例也不是负例的anchors对训练目标没有贡献

#### Loss Function:

可以看出是`cls loss`与`reg loss`的和
$$
L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p_i^\star)\\
+\lambda\frac{1}{N_{reg}}\sum_ip_i^\star L_{reg}(t_i,t_i^\star)
$$
$i$ is the index of an anchor in a mini-batch and $p_i$ is the predicted probability of anchor $i$ being an object. The ground-truth label $p_i^\star$ is 1 if the anchor is positive, and is 0 if the anchor is negative. $t_i$ is a vector representing the 4 parameterized coordinates of the predicted bounding box, and $t_i^\star$ is that of the ground-truth box associated with a positive anchor.

回归 reg loss利用的是smooth L1 $L_{reg}(t_i,t_i^\star)=R(t_i-t_i^\star)$ 以及 $p_i^\star L_{reg}$  means the regression loss is activated only for positive anchors ($p_i^\star=1$) and is disabled otherwise ($p_i^\star = 0$). 只计算正例的回归损失，不在乎反例的回归损失

这里$\lambda$默认设置为10

#### anchors的变形和bounding box回归相同

## ROI Pooling



## 之后预测

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnn5.jpg)

## Reference

[Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF) 这个和YOLO的不同，似乎是利用导包等实现的，没有太具体的实现过程，都封装好了，可能还需要再阅读caffe的源码才行。

[fasterRCNN](https://github.com/rbgirshick/py-faster-rcnn)

[像玩乐高一样拆解Faster R-CNN：详解目标检测的实现过程](https://www.jiqizhixin.com/articles/2018-02-23-3)