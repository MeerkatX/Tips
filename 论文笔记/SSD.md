# SSD: Single Shot Multi Box Detector

RCNN系列是proposal+classification的方式，two-stage.

YOLO系列将边框和打分合在一起，one-stage

SSD也是one-stage，SSD是在faster RCNN之后

## 简介

Our approach, named SSD, discretizes the output space of bounding boxes into a set of **default boxes** *（这里的 d box 就和 faster RCNN 中的 anchors 类似）* over different **aspect ratios and scales** per feature map location*（因为是多尺度的，分别在 $38\times38​$，$19\times19​$，$10\times10​$，$5\times5​$，$3\times3​$，$1\times1​$ 这几个尺度上进行回归预测，所以相对的放缩，长宽比每一层相应的调整）*. At prediction time,the network generates **scores** for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.（在预测是，网络生成每个种类的分数，然后做回归调整更匹配这个目标的形状）

上面是原文的简介。

## 网络结构

![img](https://github.com/MeerkatX/Tips/blob/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/SSD.png)

以及

![img](https://pic2.zhimg.com/v2-57a84a027f8ab07209991d850280ac83_r.jpg)

## Convolutional predictors for detection

​        每个添加的特征层（或可选的基础网络的现有特征层）可以使用一组卷积滤波器产生固定的预测集合。这些在图2中SSD网络架构顶部已指出。对于具有p个通道的大小为m×n的特征层，使用3×3×p卷积核卷积操作，产生类别的分数或相对于默认框的坐标偏移。在每个应用卷积核运算的m×n大小位置处，产生一个输出值。边界框偏移输出值是相对于默认框测量，默认框位置则相对于特征图（参见YOLO [5]的架构，中间使用全连接层而不是用于该步骤的卷积滤波器。

## Training objective

$$
L(x,c,l,g)=\frac{1}{N}(L_{conf}(x,c)+\alpha L_{loc}(x,l,g))
$$

$x^p_{ij}=\{1,0\}$为第 i 个default box和第 j 个的类别为P的实际框相匹配，可以匹配多个使得$\sum_ix^p_{ij}\geq1$

N是匹配的d box总数。N=0，loss=0
$$
L_{loc}(x,l,g)=\sum^N_{i\in Pos}\sum_{m\in {cx,cy,w,h}}x^k_{ij}smooth_{L1}(l^m_i-\hat g^m_j)\\
\hat g_j^{cx}=(g^{cx}_j-d^{cx}_i)/d^w_i\\
\hat g_j^{cy}=(g^{cy}_j-d^{cy}_i)/d^h_i\\
\hat g^w_j=\log\big(\frac{g^w_j}{d^w_j}\big)\\
\hat g^h_j=\log\big(\frac{g^h_j}{d^h_j}\big)
$$
b-box回归和RCNN的一样

confidence loss是`softmax` loss也就是交叉熵
$$
L_{conf(x,c)}=-\sum^N_{i\in Pos}x^p_{ij}log(\hat c^p_i)-\sum_{i\in Ncg}log(\hat c^0_i)\\
\hat c^p_i=\frac{\exp(c^p_i)}{\sum_p\exp(c^p_i)}
$$

## Choosing scales and aspect ratios for default boxes

Suppose we want to use **m feature maps** for prediction. The **scale** of the **default boxes** for each feature map is computed as:计算m个特征图的default box的缩放尺度，可以结合代码理解
$$
s_k=s_{min}+\frac{s_{max}-s_{min}}{m-1}(k-1),k\in [1,m]
$$

其中$s_{min}=0.2,s_{max}=0.9$。 meaning the lowest layer has a scale of 0.2 and the highest layer has a scale of 0.9, and all layers in between are regularly spaced. 

长宽比为$\alpha_r\in\{1,2,3,\frac{1}{2}\frac13\}$ ，其中width为$w_k^a=s_k\sqrt{a_r}$ 以及 $h^a_k=\frac{s_k}{\sqrt{a_r}}$，其中把aspect ratio = 1的情况多加一个$s’_k=\sqrt{s_ks_{k+1}}$，所以连着前面的五个，一共有6种长宽比。

设置每个default box 的中心为$(\frac{i+0.5}{|f_k|},\frac{j+0.5}{f_k})​$其中$|f_k|​$是第K个的正方形feature map的大小，随后截取默认框坐标使其始终在[0，1]内，$i,j\in[0,|f_k|]​$。

## Hard negative mining

这个在YOLO中有相关介绍

以及为什么要用这个其实是因为负例太多，正例太少，梯度下降的话冲淡正例的优化，所以只用了一部分负例和全部正例进行训练，大约比例为1:3

## Data augmentation

这个也是与YOLO中类似

- Use the entire original input image.

- Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3, 0.5, 0.7, or 0.9. 

- Randomly sample a patch. 

## Reference

[SSD在训练什么](https://zhuanlan.zhihu.com/p/29410169)

[SDD-TensorFlow](https://github.com/balancap/SSD-Tensorflow)

[目标检测 -- SSD (tensorflow 版) 逐行逐句解读](https://blog.csdn.net/qq1483661204/article/details/79776065)

## Code

下面是SSD网络整体结构的定义：

```python
# num_classes=21 共20类加背景
# feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
# feat_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
# anchor_steps=[8, 16, 32, 64, 128, 256, 512] 这里 8*64=512 16*32=512等，特征图的缩放倍数
# 是对应的vgg500
# anchor_size_bounds=[0.10, 0.90]
"""
由上面的anchor_size_bounds来计算anchor_sizes 0.1*512=51.2 ... 0.9*521=480.8
anchor_sizes=[(20.48, 51.2),
    (51.2, 133.12),
    (133.12, 215.04),
    (215.04, 296.96),
    (296.96, 378.88),
    (378.88, 460.8),
    (460.8, 542.72)]
长宽比，与论文不同的是，最后两层也只用了两个尺度
anchor_ratios=[[2, .5],
    [2, .5, 3, 1./3],
    [2, .5, 3, 1./3],
    [2, .5, 3, 1./3],
    [2, .5, 3, 1./3],
    [2, .5],
    [2, .5]]
"""
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):
    """
    SSD net definition.
    """
    # SDD前几层是VGG-16的特征提取层
    # End_points collect relevant activations for external use.
    end_points = {}# 这里，因为要对很多层特征图进行bbox回归，分类，所以需要保存下来feature map
    with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        # slim.repeat应该是重复slim.conv2d 两次 卷积核为3*3*64
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        # Additional SSD blocks.这里开始添加SSD的块
        # Block 6: let's dilate the hell out of it!
        # 这里rate即扩展卷积或带孔卷积，指定用于atrue的扩张率，使用了atrous algorithm
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        # Block 7: 1x1 conv. 
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block12'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [4, 4], scope='conv4x4', padding='VALID')
            # Fix padding to match Caffe version (pad=1).
            # pad_shape = [(i-j) for i, j in zip(layer_shape(net), [0, 1, 1, 0])]
            # net = tf.slice(net, [0, 0, 0, 0], pad_shape, name='caffe_pad')
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        # 这里对['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
        # 进行预测定位
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                # 这里预测每个block对应着相应的anchor_size，anchor_ratios
                p, l = ssd_vgg_300.ssd_multibox_layer(end_points[layer],
                                                      num_classes,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      normalizations[i])
            predictions.append(prediction_fn(p))
            # prediction_fn(p)即softmax
            logits.append(p)
            localisations.append(l)# 保存预测框

        return predictions, localisations, logits, end_points
```

紧接着上面 `ssd_multibox_layer`

```python
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)
 
    # Location.定位，anchors的数量*4，即xc,yc,w,h
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')# 在特征图上进行3*3*num_anchors*4的卷积进行计算
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.和上面相似
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred
```

这个函数将NHWC或NCHW切换为NHWC 估计这里主要是`caffe`用的是NCHW需要切换，但`tensorflow`不用

```python
@add_arg_scope
def channel_to_last(inputs,
                    data_format='NHWC',
                    scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a single output format whatever the input data format.
    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    Return:
      Input in NHWC format.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net
```

有关`atrous algorithm`以及`slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')` 中的`rate` 见下图：

![img](https://github.com/MeerkatX/Tips/blob/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/atrousalgorithm.jpg)