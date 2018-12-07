# 3D RoI-aware U-Net for Accurate and Eﬃcient Colorectal Tumor Segmentation

偶尔看到的一个3D分割的论文，决定看一看，提到了很多重要的问题，比如3D分割的GPU显存不足，以及提取ROI，如何改造网络，以及是该分成几部分进入网络还是整个进入

## 网络结构：

![net](https://github.com/MeerkatX/Tips/blob/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/3DRunet1.png)

基于ResNet+U-net的3D改造，在encoder阶段接受全图，在F3的特征图提取ROI区域，并且扩展到F2,F1以用来进行之后类似U-net的特征融合。

原文：we adopt a variation of ResBlock formulated 3D U-Net’s encoder, called Global Image Encoder, to process whole image volumes without dividing them into context-limited small parts. 

以及Global Image Encoder for RoI Localization:  the encoder is trained to predict **down-sampled segmentation masks** from global images.  the encoder is trained towards **Dice loss**.

 Then we perform **connectivity analysis** to compute desired bounding boxes. To make up for potential bounding box undersize due to the coarseness of low resolution prediction, the bounding boxes computed are practically extended to 1.5× or 2× of its original size or to an over-designed cube of ﬁxed size d×h×w (e.g. 24×96×96) voxels along the Z,Y and X axis. 

之后是ROI cropping layer：用来将感兴趣区域直接切割下来，如图。

切割之后上采样反卷积，进行decoder来分割。

## Dice-based Multi-task Hybrid Loss Function

文章中利用了以下loss函数，以及在之前U-net分割中，我也用了但是并不是很理解的Dice loss

文章中的loss：
$$
L_d(P,G)=1-2\times \frac{\sum^N_{i=1}p_ig_i+\epsilon}{\sum^N_{i=1}p_i+\sum^N_{i=1}g_i+\epsilon}
$$
其中$p_i$是预测值，$g_i$ 是真值, $\epsilon$ 是松弛值，一般=1

对$p_k$求偏导：
$$
\frac{\partial L_d(P,G)}{\partial p_k}=-2\times \frac{\sum^N_{i=1}p_ig_i-g_k\sum^N_{i=1}(p_i+g_i)}{[\sum^N_{i=1}(p_i+g_i)]^2}
$$


## Dice loss

dice loss 是V-Net于2016年提出的用于图像分割的一种loss。

其中的具体解释可以参考：

[What is "Dice loss" for image segmentation?](https://dev.to/andys0975/what-is-dice-loss-for-image-segmentation-3p85)

以下摘录一部分：

查准，查全率：
$$
Precison=\frac{TP}{TP+FP}\\
Recall=\frac{TP}{TP+FN}
$$
以及F1 score便是想以相同权重β=1)的**Harmonic mean(调和平均)**去整合这两个指标：
$$
\frac{1}{F^1}=\frac{1}{Precison}+\frac{1}{Recall}\to F^1=\frac{2PR}{P+R}\to F^1=\frac{2TP}{2TP+FP+FN}
$$
之后是Sorensen-dice coefficient:
$$
QS=\frac{2|X\cap Y|}{|X|+|Y|}=\frac{2TP}{2TP+FP+FN}
$$
*QS*是**Quotient of Similarity**(相似商)，就是coefficient的值，只会介于**0～1**。Image segmentation中，模型分割出的mask就是影像的**挑选总数**，专家标记的mask就是**正确总数**。对应到公式便可知**挑选总数(TP+FP)**和**正确总数(TP+FN)**分別就是*X*和*Y*，交集便是TP，可见Dice coefficient等同**F1 score**，直观上是计算X与Y的相似性，本质上则同時隐含Precision和Recall两指标。

**Dice loss**其实就是它的颠倒。当coefficient越高，代表分割結果与标准答案相似度越高，而模型則是希望用**求极小值**的思维去训练比较可行，因此常用的Loss function有 **"1-coefficient"** 或 **"-coefficient"**。

```python
smooth = 1.#Laplace smoothing可以減少Overfitting
def dice_loss(y_pred, y_true):
    product = nd.multiply(y_pred, y_true)
    intersection = nd.sum(product)
    coefficient = (2.*intersection +smooth) / (nd.sum(y_pred)+nd.sum(y_true) +smooth)
    loss = 1. - coefficient
    # or "-coefficient"
    return(loss)
```

## Reference

[3D-RU-Net github](https://github.com/huangyjhust/3D-RU-Net)