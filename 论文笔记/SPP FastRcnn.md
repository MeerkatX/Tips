# Fast RCNN



## Fast R-CNN architecture

A Fast R-CNN network takes as input an entire image and a set of object proposals. The network ﬁrst processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map. 

Then, for each object proposal a region of interest (RoI) pooling layer extracts a ﬁxed-length feature vector from the feature map. 

Each feature vector is fed into a sequence of fully connected (fc) layers that ﬁnally branch into two sibling output layers: one that produces soft-max probability estimates over K object classes plus a catch-all “background” class and another layer that out puts four real-valued numbers for each of the K object classes. Eachsetof4valuesencodesreﬁned bounding-box positions for one of the K classes. 

## RoI pooling

The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a ﬁxed spatial extent of H ×W (e.g., 7×7).

Each RoI is deﬁned by a four-tuple (r,c,h,w) that speciﬁes its top-left corner (r,c) and its height and width (h,w).

RoI max pooling works by dividing the h×w RoI window into an H × W grid of sub-windows of approximate size h/H ×w/W and then max-pooling the values in each sub-window into the corresponding output grid cell. 

## 为什么SPP无法fine-tune卷积层：

The root cause is that back-propagation through the SPP layer is **highly inefﬁcient** when each training sample (i.e. RoI) comes from a **different image**, which is exactly how R-CNN and SPP-net networks are trained.

这里说的主要原因是**过于低效**，低效的原因是训练取样源自于不同的图像。根据知乎的一个回答来说，是先将所有的图片的区域提议保存，然后随机选128个区域提议进行训练（因此可能128个区域源自于128张图片，就都要传入网络进行计算。）

这里还有一点按照自己的理解来看是SPP将整个图片传入进行特征提取，然后对RoI进行特征金字塔池化，按照上面说的就是随机选128 RoI，然后如果是一张图片中的几个RoI的话，就利用特征映射，一次获得这几个RoI的特征图。

所以fast RCNN解决的方法是将minibatch SGD 是随机选择两张图片，再从两张图片中随机选128个RoI。

The inefﬁciency stems from the fact that each RoI may have a very large receptive ﬁeld,often spanning the entire input image. Since the forward pass must process the entire receptive ﬁeld, the training inputs are large (often the entire image).

## Loss

$$
L(p,u,t^u,v)=L_{cls}(p,u)+\lambda[u\geq1]L_{loc}(t^u,v)
$$

其中$L_{cls}(p,u)=-\log p_u$  log loss for true class u
$$
L_{loc}(t^u,v)=\sum_{i\in\{x,y,w,h\}}smooth_{L_1}(t^u_i-v_i)
$$
The second task loss, $L_{loc}$, is deﬁned over a tuple of true bounding-box regression targets for class u, v = $(v_x,v_y,v_w,v_h)$, and a predicted tuple $tu = (t^u_x,t^u_y,t^u_w,t^u_h)$, again for class u. The Iverson bracket indicator function $[u ≥ 1]$ evaluates to 1 when $u ≥ 1$ and 0 otherwise. By convention the catch-all background class is labeled $u = 0$. 
$$
smooth_{L_1}(x)=
\begin{cases}
0.5x^2,& \text{if|x|<1}\\
|x|-0.5,& \text{otherwise}
\end{cases}
$$
对于为什么用L1 loss ：

is a robust L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPP-net.  When the regression targets are unbounded, training with L2 loss can require careful tuning of learning rates in order to prevent exploding gradients.（当回归目标无限时，L2损失训练可能需要仔细调整学习速率以防止梯度爆炸）

$\lambda$用来平衡两边loss，分类和回归loss。 We normalize the ground-truth regression targets vi to have zero mean and unit variance. All experiments use λ = 1. 

## ROI pooling的反向传播

对于每个输入变量$x_i$
$$
\frac{∂L}{∂x_{i}}=\sum_r\sum_j[i=i\star(r,j)]\frac{∂L}{∂y_{rj}}
$$
对于**每个小批量RoI** $r$ 和对于每个池化输出单元 $y_{rj}$ ，如果 $i$ 是 $y_{rj}$ 通过最大池化选择的argmax，则将这个偏导数$\frac{∂L}{∂y_{rj}}$积累下来。在反向传播中，偏导数$\frac{∂L}{∂y_{rj}}​$已经由RoI池化层顶部的层的反向传播函数计算

## Truncated SVD for faster detection

TODO 这里需要再看看

## Reference

[为什么SPP-Net无法fine-tune卷积层?](https://www.zhihu.com/question/66283535)