# Deep learning-based computer-aided diagnosis system for localization and diagnosis of metastatic lymph nodes on ultrasound: A pilot study 

这个论文没有太多介绍仔细的介绍如何实现的，只是大概说了以下

## Materials and Methods 

#### Patients and Datasets 

From January 2008 to December 2015,612 lymph nodes form 604 patients(293 benign lymph nodes from 293 patients, 319 metastatic lymph nodes from 311 patients) were consecutively examined **训练数据和验证数据**

- 263 benign（良性） 286 metastatic（恶性）为训练数据

- 30 benign 33 metastatic 为验证数据

From January to December 2016, we also enrolled 

- 200 lymph nodes (100 benign and 100 metastatic lymph nodes) as a test data set. 为测试数据

分别利用了手术前后的淋巴结。

#### Data pre-processing

数据预处理 因为有blood vessels（血管） and adipose（脂肪） and muscle tissues（肌肉组织）等噪声的存在，所以用了data augmentation

- 数据扩增的方法：

the image angle was set randomly within ±15º （图像角度在±15°内随机设置）

- 预处理：

resized all of the augmented images to 224 × 224 pixel之后就输入到CNN模型。 

As this task could be performed without requiring label information, we also applied it to the validation and test sets used for model evaluation. 

#### Deep neural network

Overall, weakly supervised learning（弱监督学习） has increasingly attracted attention because it can be used to detect the location of a meaningful object from an **attention heatmap** in the **absence of location information**.

这里参考了的文献 （得仔细看一下）

- On learning to localize objects with minimal supervision
- Grad-cam: Visual explanations from deep networks via gradient-based localization
- Constrained convolutional neural networks for weakly supervised segmentation Proceedings of the IEEE International Conference on Computer Vision

Zhou et al. proposed a method called **class activation mapping** (CAM), which uses **global average pooling** (GAP), and showed that objects can be **located clearly without location information**

使用的是CNN-GAP去定位和分类

#### Training and diagnostic performance evaluation

- pre-trained model to Image-Net for the initialization of network parameters 同样也是利用image-net去预训练初始化网络参数

- 学习率  set the **learning rate** to **0.001** and decreased it by a factor of **10** 
-  we **cropped the important region**, defined as the area where the CNN-GAP network exhibited the **maximum activation value**, and **set it as the lymph node location**. 

-  **validation** and **test** data sets were also amplified（放大） 

#### localization of the lymph node

- we placed an attention heatmap drawn using GAP on the lymph node to allow us to infer the location 
-  the model focuses on the medically and biologically important parts of an input image to classify the node as benign or malignant.
-  we cropped the important region containing the maximum CNNGAP network activation value and drew a rectangle on the US image to show the location of the lymph nodes. 

## Reference

## Learning Deep Features for Discriminative Localization

#### GAP (Global average pooling)

即对整个特征图的其中一个channel求平均值。如$8\times8\times256$，那么最后得到256个数

#### CAM (Class Activation Mapping)

![img](https://github.com/MeerkatX/Tips/blob/master/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/imgs/cam.png)

第k个特征图是$F_k=\sum_{x,y}f_k(x,y)$
$$
S_c=\sum_kw^c_k\sum_{x,y}f_k(x,y) \\
=\sum_{x,y}\sum_{k}w^c_k f_k(x,y)
$$
We deﬁne Mc as the class activation map for class c, where each spatial element is given by 
$$
M_c(x,y)=\sum_kw_c^cf_k(x,y)
$$
其实就是将最后一层学习到的权重$w^c_k$与之对应的特征图$f_k​$乘起来，然后将所有的特征图相加，再把它强行resize到原图尺寸。（By simply upsampling the class activation map to the size of the input image, we can identify the image regions most relevant to the particular category. ）

