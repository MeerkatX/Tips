# R-CNN

#### 简要步骤

1. 输入测试图像
2. 利用选择性搜索Selective Search算法在图像中从下到上提取2000个左右的可能包含物体的候选区域Region Proposal
3. 因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN，将CNN的fc7层的输出作为特征
4. 将每个Region Proposal提取到的CNN特征输入到SVM进行分类

#### Introduction

需要解决的两个问题：localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data

有大概三种设想的解决方案：

 One approach frames localization as a regression problem.将定位看作回归问题

 An alternative is to build a sliding-window detector.另一种是建立滑动窗口检测(计算量太大舍弃)

recognition using regions 区域提议 RCNN使用的方法

 our method generates around 2000 category-independent region proposals for the input image , extracts a ﬁxed-length feature vector from each proposal using a CNN, and then classiﬁes each region with category-speciﬁc linear SVMs. We use a simple technique (afﬁne image warping) to compute a ﬁxed-size CNN input from each region proposal, regardless of the region’s shape.

#### 特点

