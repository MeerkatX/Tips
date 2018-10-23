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

#### 代码如下：

假设通过了vgg16的最后的conv5 3*3 得到feature map之后：

```python
rpn = slim.conv2d(net_conv, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")#这里利用卷积来实现滑动窗口，利用3*3*512的卷积核滑动
```

通过上面得到 $\times3\times512$ 的feature map

```python
 # shape = (1, ?, ?, 18) , 其中，batchsize=1
 # 之后再利用[1*1*2*9]的卷积核做 前景背景的预测 (1,特征图宽，特征图高，18)
rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                            weights_initializer=initializer,
 # change it so that the score has 2 as its channel size
 # shape = (1, ?, ?, 2)
rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
 # shape = (1, ?, ?, 2)
 # 这里利用sofmax来做 其中softmax_layer这个函数如果接收到rpn_cls_prob的话会reshape后再tf.nn.softmax
 # 然后返回值会再reshape回去
rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
 # shape = (?,) 这里找到每个窗口含有各类的最大值
rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
 # shape = (1, ?, ?, 18) 
rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
 # shape = (1, ?, ?, 36) 这里同样利用1*1*36的卷积做回归 其中形状应该就是 (1,特征图宽，特征图高，36)
rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                            weights_initializer=initializer,
                            padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
```

由此可以得到各`anchors` 和 `softmax` 的分数，传到下一层的`ROI pooling`层对感兴趣的区域进行池化特征提取，之后就和`fast rcnn`一样，进行`softmax`和回归

如果是训练阶段的话，传入`ground truth`和种类进行训练

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

在Fast R-CNN网络中，原始图片经过多层卷积与池化后，得到整图的feature map。而由selective search产生的大量proposal经过**映射**可以得到其在feature map上的映射区域（ROIs），这些ROIs即作为ROI Pooling层的输入。

ROI Pooling时，将输入的h * w大小的feature map**分割成H * W**大小的子窗口（每个子窗口的大小约为h/H，w/W，其中H、W为超参数，如设定为7 x 7），对每个子窗口进行**max-pooling**操作，得到**固定输出**大小的feature map。而后进行后续的全连接层操作。

同理在Faster RCNN中就是将RPN网络中的区域进行ROI池化。

```python
def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # 得到归一化的bbox坐标（相对原图的尺寸进行归一化）
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = cfg.POOLING_SIZE * 2
        # 裁剪特征图，并resize成相同的尺寸
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
        # 进行标准的max pooling
    return slim.max_pool2d(crops, [2, 2], padding='SAME')
```



## 之后预测

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/fasterrcnn5.jpg)

## Reference

[tf-faster-RCNN](https://github.com/endernewton/tf-faster-rcnn)

[fasterRCNN](https://github.com/rbgirshick/py-faster-rcnn)

[像玩乐高一样拆解Faster R-CNN：详解目标检测的实现过程](https://www.jiqizhixin.com/articles/2018-02-23-3)

[tf.image.crop_and_resize](https://blog.csdn.net/m0_38024332/article/details/81779544)

`voc`图像对应的xml信息：如果是 `xmin` `xmax`的方式来说的话应该需要修改成`xcenter` `ycenter`，简单的$w=x_{max}-x_{min}$ 以及 $h=y_{max}-y_{min}$ ，$x_{center}=w/2+偏移$，$y_{center}=h/2+偏移$

```xml
<annotation>  
    <folder>VOC2007</folder>                             
    <filename>2007_000392.jpg</filename>                               //文件名  
    <source>                                                           //图像来源（不重要）  
        <database>The VOC2007 Database</database>  
        <annotation>PASCAL VOC2007</annotation>  
        <image>flickr</image>  
    </source>  
    <size>                                               //图像尺寸（长宽以及通道数）                        
        <width>500</width>  
        <height>332</height>  
        <depth>3</depth>  
    </size>  
    <segmented>1</segmented>                                   //是否用于分割（在图像物体识别中01无所谓）  
    <object>                                                           //检测到的物体  
        <name>horse</name>                                         //物体类别  
        <pose>Right</pose>                                         //拍摄角度  
        <truncated>0</truncated>                                   //是否被截断（0表示完整）  
        <difficult>0</difficult>                                   //目标是否难以识别（0表示容易识别）  
        <bndbox>                                                   //bounding-box（包含左上角和右下角xy坐标）  
            <xmin>100</xmin>  
            <ymin>96</ymin>  
            <xmax>355</xmax>  
            <ymax>324</ymax>  
        </bndbox>  
    </object>  
    <object>                                                           //检测到多个物体  
        <name>person</name>  
        <pose>Unspecified</pose>  
        <truncated>0</truncated>  
        <difficult>0</difficult>  
        <bndbox>  
            <xmin>198</xmin>  
            <ymin>58</ymin>  
            <xmax>286</xmax>  
            <ymax>197</ymax>  
        </bndbox>  
    </object>  
</annotation> 
```

