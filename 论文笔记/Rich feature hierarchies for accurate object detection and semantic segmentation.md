# R-CNN

### 简要步骤

1. 输入测试图像
2. 利用选择性搜索Selective Search算法在图像中从下到上提取2000个左右的可能包含物体的候选区域Region Proposal
3. 因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN，将CNN的fc7层的输出作为特征
4. 将每个Region Proposal提取到的CNN特征输入到SVM进行分类

### Introduction

需要解决的两个问题：localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data

有大概三种设想的解决方案：

 One approach frames localization as a regression problem.将定位看作回归问题

 An alternative is to build a sliding-window detector.另一种是建立滑动窗口检测(计算量太大舍弃)

recognition using regions 区域提议 RCNN使用的方法

 our method generates around **2000** category-independent **region proposals** for the input image , extracts a **ﬁxed-length** feature vector from each proposal using a CNN, and then classiﬁes each region with category-speciﬁc linear SVMs. We use a simple technique (**afﬁne image warping**) to compute a ﬁxed-size CNN input from each region proposal, **regardless of the region’s shape**.

另一个问题是labeled data is scarce and the amount currently available is insufficient for training a large CNN.

解决方案是use unsupervised pre-training,followed by supervised fine-tuning，预训练一个模型在再调整其中的参数

### Object detection with R-CNN 

模型分为三部分:

The ﬁrst generates category-independent **region proposals**. 

The second module is a large **convolutional neural network** that extracts a ﬁxed-length feature vector from each region. 

The third module is a set of class-speciﬁc linear **SVM**s.

1. Module design

   - Region proposals  **selective search**

   - Feature extraction  

     extract a **4096-dimensional feature** vector from each region proposal

     Features are computed by forward propagating a **mean-subtracted 227 × 227 RGB** image through **ﬁve convolutional layers** and **two fully connected layers**. 

     warp all pixels in a tight bounding box around it to the required size
2. Test-time detection

- 一开始，系统先用selective search提取2000个候选区域，并将其warp到277\*277大小，进入CNN提取特征，并用SVM分类。最后，再用 greedy non-maximum suppression 把那些高度重叠的框剔除。

3. Training

- Supervised pre-training：先将CNN在ILSVRC 2012上进行预训练（with image-level annotations (i.e., no bounding box labels))
- Domain-specific fine-tuning：微调过程，以**0.001**的学习速率进行**SGD**训练。对某个分类**只要IOU>0.5**就视该边框为正值。每次SGD迭代都采样**38个正边框和96个背景**。
- Object category classifiers：对某个分类，高IOU和IOU都很好区分，但IOU处于中值时则很难定义生成的候选框是否包含了该物体。设定了一个**阈值0.3**，低于它的一律视为背景（负数）。另外，每个分类都优化一个SVM。由于负样本很多，因此还采用了**hard negative mining**方法 

### Visualization,ablation,and modes of error 

1. Performance layer-by-layer, with ﬁne-tuning一层一层的微调

     which suggests that the **pool5 features learned from ImageNet are general** 池化层得到的特征是通用的 and that most of the improvement is gained from learning domain-speciﬁc non-linear classiﬁers on top of them.

2. Bounding box regression

     we train a linear regression model to predict a new detection window given the pool5 features for a selective search region proposal. 加入bounding box回归能够提高准确度
$$
P^i=(P^i_x,P^i_y,P^i_w,p^i_h)
$$

​       映射到

$$
G=(G_x,G_y,G_w,G_h)
$$
​       通过参数$(d_x,d_y,d_w,d_h)$平移加尺度放缩 (至于为什么会是这种形式可以参考下面给出的博客)
​       平移$(\Delta{x},\Delta{y})$，$\Delta{x}=P_wd_x(P),\Delta{y}=P_hd_y(P)​$
$$
\hat{G_x}=P_wd_x(P)+P_x
$$

$$
\hat{G_y}=P_hd_y(P)+P_y
$$

​       尺度缩放 $(S_w,S_h)$,$S_w=\exp(d_w(P)),S_h=\exp(d_h(P))$
$$
\hat{G_w}=P_w\exp{(d_w(P))}
$$

$$
\hat{G_h}=P_h\exp{(d_h(P))}
$$

​       真正需要的平移量和尺度缩放
$$
t_x=(G_x-P_x)/P_w
$$

$$
t_y=(G_y-P_y)/P_h
$$

$$
t_w=\log(\frac{G_w}{P_w})
$$

$$
t_h=\log(\frac{G_h}{P_h})
$$

​     计算d与t之间的L2 loss 其中$d_\star(P)={\omega}^T_{\star}\Phi_5(P)$

3. 分别观察了没有微调的，和微调的pool5 fc6 fc7。最后得到经过 fine-tuning 的 fc7 加上BB回归能得到最好的准确度

看图比较好。

### 结论

The ﬁrst is to apply high-capacity convolutional neural networks to bottom-up region proposals in order to localize and segment objects.  The second is  a paradigm  for training  large  CNNs  when  labeled   training data is scarce. 

### Reference

[selective search](https://zhuanlan.zhihu.com/p/27467369)

[RCNN](https://zhuanlan.zhihu.com/p/27473413)

[Bounding Box回归](https://blog.csdn.net/zijin0802034/article/details/77685438)

