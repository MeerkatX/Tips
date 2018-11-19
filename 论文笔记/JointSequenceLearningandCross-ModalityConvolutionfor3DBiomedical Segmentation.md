# Joint Sequence Learning and Cross-Modality Convolution for 3D Biomedical Segmentation

## 简介：

CVPR 2017论文，3D分割医学生物图像的一种方法。

这个文章是对脑瘤的分割，利用MRI图像即核磁共振，其中扫描后得到4种图来识别不同的组织，所以将这些特征做了cross-modality的提取。之后扫描后的图片是序列图片，因此采用了convolution LSTM来获取空间和时序上的特征。一个端到端的方法。

文章对比了几种不同的方法来分割3D的医学图像：

1. U-Net分别对每一帧来进行分割然后穿起来做序列的，但是明显丢失了时序上的特征。
2. 利用3D卷积来分割，但是需要很大数据量的datasets而且容易过拟合。
3. 提出了自己的融合多种特征以及利用Convolution LSTM来进行分割。

大概模型结合了这几个部分：

1. multi-modal encoder, cross-modality convolution and convolutional LSTM.
2. The slices from different modalities are stacked together by the depth values. 
3. Then, they pass through different CNNs in the multi-modal encoder (each CNN is applied to a different modality) to obtain a semantic latent feature representation  Latent features from multiple modalities are effectively aggregated by the proposed cross-modality convolution layer.
4.  Then,we leverage convolutional LSTM to better exploit the spatial and sequential correlations of consecutive slices. 

![img](https://raw.githubusercontent.com/MeerkatX/Tips/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/crossmod.png)

未完待续…