# U-Net: Convolutional Networks for Biomedical Image Segmentation

1. **下采样+上采样：Convlution + Deconvlution／Resize**
2. **多尺度特征融合：特征逐点相加／特征channel维度拼接**
3. **获得像素级别的segement map：对每一个像素点进行判断类别**

## Abstract

we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more eﬃciently. 

The architecture consists of a **contracting path 收缩路径** to capture context and a **symmetric expanding path 对称的扩张路径** that enables precise localization. 

## Network Architecture 网络结构：

![unetarch](https://github.com/MeerkatX/Tips/blob/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/unet.png)

#### 卷积收缩路径：

The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of **two 3x3 convolutions** (**unpadded** convolutions), each followed by a rectiﬁed linear unit (ReLU) and a **2x2 max pooling** operation with **stride 2** for down-sampling. 

At each down-sampling step we **double** the number of **feature channels**.

#### 转置卷积 扩展路径：

Every step in the expansive path consists of an up-sampling of the feature map followed by a **2x2 convolution** (“up-convolution”) that **halves the number of feature channels**（每次上采样变成一半的特征通道数）, a concatenation with the correspondingly cropped feature map from the contracting path（之后加上对应的剪切过的特征图在后面）, and two 3x3 convolutions, each followed by a ReLU. 

The cropping is necessary due to the loss of border pixels in every convolution. 

与FCN逐点相加不同，U-Net采用将特征在channel维度拼接在一起，形成更“厚”的特征

**语义分割网络在特征融合时也有2种办法：**

1.  **FCN式的逐点相加，对应`caffe`的`EltwiseLayer`层，对应`tensorflow`的`tf.add()`**
2.  **U-Net式的channel维度拼接融合，对应`caffe`的`ConcatLayer`层，对应`tensorflow`的`tf.concat()`**

#### 最后一层 1x1 的卷积做`softmax`对每个像素分类

At the ﬁnal layer a **1x1** convolution is used to map each 64 component feature vector to the desired number of classes. 

最后再经过两次卷积，达到最后的heatmap，再用一个1X1的卷积做分类，这里是分成两类，所以用的是两个神经元做卷积，得到最后的两张heatmap,例如第一张表示的是第一类的得分（即每个像素点对应第一类都有一个得分），第二张表示第二类的得分heatmap,然后作为softmax函数的输入，算出概率比较大的softmax类，选择它作为输入给交叉熵进行反向传播训练

#### CODE：

```python
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
	# 这里进行上采样操作，利用 2*2*1024*512 的卷积核
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # 这里进行合并操作，将drop4的特征图加入到up6的后面
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    # 之后继续进行两个卷积操作
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # 最后利用 1*1的卷积进行分类,这里分为 1 类，如果对于细胞来说，做边界分割的话
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
```

## Training:

The energy function is computed by a pixel-wise **soft-max** over the **ﬁnal feature map** combined with the **cross entropy loss function**. 

#### 交叉熵函数：

The cross entropy then penalizes(惩罚)  **at each position** the deviation of $p_{l(x)}(x)$ 
$$
E=\sum_{x\in \Omega}w(X)\log(p_{l(x)}(X))
$$
where $l : Ω → \{1,...,K \}$ is the **true label of each pixel** and $w : Ω → \mathbb{R}$ is a weight map that we introduced to give some pixels more importance in the training. 

为了凸显某些像素点更加重要，我们在公式中引入了 $w(X)$

我们对每一张标注图像预计算了一个权重图，来补偿训练集中每类像素的不同频率，使网络更注重学习相互接触的细胞之间的小的分割边界。权重图计算公式如下：
$$
w(X)=w_c(X)+w_0\cdot\exp(-\frac{(d_1(x)+d_2(x))^2}{2\delta^2})
$$
这里$w_c$用于平衡类别出现频率的权重( $w_c : Ω → R$ is the weight map to balance the class frequencies)，$d_1$代表到最近细胞的边界的距离，$d_2$代表到第二近的细胞的边界的距离。设定$w_0=10,\delta\approx5$像素。

#### 初始权重：

应该调整初始权重，使网络中的每个特征映射具有近似的单位方差。权重进行高斯分布初始化，分布的标准差为$\sqrt{\frac{2}{N}}$ 。

N为每个神经元的输入节点数量。例如，对于一个上一层是64通道的3\*3卷积核来说，N=9\*64。

## Data Augmentation

 we use excessive data augmentation by applying **elastic deformations**弹性变形 to the available training images. 

我们使用随机位移矢量在粗糙的3*3网格上(random displacement vectors on a coarse 3 by 3 grid)产生平滑形变(smooth deformations)。

位移是从10像素标准偏差的高斯分布中采样的。然后使用bicubic插值计算每个像素的位移。在contracting path的末尾采用drop-out 层更进一步增加数据。

## Reference

[U-Net翻译](https://zhuanlan.zhihu.com/p/37496466)

[图像语义分割入门+FCN/U-Net网络解析](https://zhuanlan.zhihu.com/p/31428783)

[unet](https://github.com/zhixuhao/unet)

[图像数据增强之弹性形变Elastic Distortions](https://zhuanlan.zhihu.com/p/46833956)