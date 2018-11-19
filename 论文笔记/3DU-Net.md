# 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation

## 简介：

基本上看起来和2D的U-Net差不多，只是做了3D。 we suggest a deep network that learns to generate dense volumetric segmentations, but only requires some annotated 2D slices for training. 

## 网络结构图：

![3dunet](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/3dunet.png)

#### 网络结构：

In the analysis path, each layer contains two 3×3×3 convolutions each followed by a rectiﬁed linear unit (ReLu), and then a 2 × 2 × 2 max pooling with strides of two in each dimension.

In the synthesis path, each layer consists of an upconvolution of 2 × 2 × 2 by strides of two in each dimension, followed by two 3 × 3 × 3 convolutions each followed by a ReLu. 

Shortcut connections from layers of equal resolution in the analysis path provide the essential high-resolution features to the synthesis path. 

In the last layer a 1×1×1 convolution reduces the number of output channels to the number of labels which is 3 in our case.

#### 问题

其中利用了的数据集以及生物专业术语看不懂。

## Training

![training](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/3dunettrain.png)

## Code:

#### 这里利用了keras单纯就网络结构来说比较简单：

```python
 # 网络结构：
 # 收缩路径
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    # 上采样路径：
    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        # 进行反卷积
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=current_layer._keras_shape[1])(current_layer)
        # 进行链接
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        # 对于每一次反卷积后进行两次卷积操作
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=current_layer, batch_normalization=batch_normalization)
    # 最后一层进行(1,1,1)的卷积利用sigmoid激活
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
```

#### 创建的卷积块：

```python
# 默认的kernel是3x3x3即在时间维度即 前一张，中间一张，后一张 三者之间进行3x3卷积
# 具体实现调用3D卷积就行了
def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    # 这里调用3D卷积就行
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
```

#### 创建的反卷积模块

```python
## 反卷积函数：
def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
```

## Reference

[3DUnetCNN](https://github.com/ellisdg/3DUnetCNN)

gitHub上利用3D U-Net分割的效果图：
![tumor_segmentation](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/tumor_segmentation_illusatration.gif)

