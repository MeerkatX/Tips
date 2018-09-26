# Deep learning 深度学习

## 摘要部分

#### machine learning 机器学习

1. identify objects in images 识别图像中目标

2. transcribe speech into text 语音转换文本

3. match news items 匹配新闻元素

4. posts or products with user's interests 根据用户兴趣提供职位产品

5. select relevant results of search 选择搜索结果

#### 传统机器学习局限性

1. 处理未加工数据(natural data in their raw form)能力有限

2. 需要构建特征提取器(feature extractor)将原始数据转化为适合的内部特征(internal representation)或特征向量(feature vector)

#### Representation learning 表示学习

1. 能自动发现检测或分类所需要的表示(特征)

2. Deep leanring是表示学习

#### Deep leanring 深度学习

1. 组合(compose)简单的非线性模型(simple but non-linear modules)转化(transform)为更高级，抽象的特征

2. 各层特征不是利用人工来标注的，而是使用通用的学习过程从数据中获得的

3. 深度学习的发展和运用


#### Supervised learning 监督学习

#### supervised learning

#### stochastic gradient descent (SGD) 随机梯度下降算法

1. This consists of showing the input vector for a few examples, computing the outputs and the errors(计算输出值和误差), computing the average gradient(平均梯度) for those examples, and adjusting the weights accordingly. The process is repeated for many small sets of examples(小的样本集) from the training set until the average of the objective function(目标函数) stops decreasing. 

2. 目前的作业(logistic regression)就用到了随机梯度下降，而linear regression用的是梯度下降(对所有样本梯度下降)

3. 利用随机小样本，计算速度快(surprisingly quickly)

#### test set 测试集
1.  generalization ability of the machine 机器的泛化能力
#### linear classifiers 线性分类器
1.  two-class linear classifier 二分类 计算特征向量的加权和 (即Wx+b)  
    a weighted sum of the feature vector 
2.  threshold 阈值 category 种类
3.  hyperplane 超平面
#### 线性分类器在图像，语音问题下的缺点
1.  insensitive(忽略) to irrelevant
    variations of the input 输入样本中的不相关元素 

2.  线性分类器或其它浅层(shallow)分类器不能区分相同背景的萨摩耶和狼

3.  non-linear
    features 非线性特征 kernel methods 核方法 Gaussian kernel 高斯核 

#### 深度学习的体系结构

1.  multilayer stack of simple modules 简单模块的多层栈，all (or most) of which are subject to learning(绝大部分的目标 是学习), and many of which compute non-linear input–output mappings(映射)
2.  multiple non-linear layers 非线性多层

## Backpropagation to train multilayer architectures 反向传播算法训练多层神经网络

#### BP算法

1. the chain
   rule for derivatives 链式求导法则

2. 核心思想

   The key insight is that the derivative (or gradient) of the objective with respect to the input of a module can be computed by working backwards from the gradient with respect to the output of that module (or the input of the subsequent module)

3. feedforward neural network 前馈神经网络  non-linear function非线性(激活)函数  rectified linear unit (ReLU) , tanh , sigmoid
$$
f(z)=\max(z,0)
$$

$$
f(z)=\tanh(z) 
$$

$$
f(z)=\frac{1}{1+e^{-z}}
$$

4.  Canadian Institute for Advanced Research (CIFAR) unsupervised learning 无监督学习


## Convolution neural networks CNN卷积神经网络

####  CNN
1.  local connections(局部链接)<br>shared weights(共享权值)<br>pooling(池化)<br>the use of many layers(多网络层)
2.  convolutional layers卷积层<br>pooling layers池化层<br>feature maps 特征图<br>filter bank 滤波器 
3.  CNN组成部分：
    1. 输入层(图片的像素矩阵) 
    2. 卷积层(对每一小块进行深入分析得到抽象程度更高的特征)
    3. 池化层(将一张分辨率较高的图片转化为分辨率较低图片)
    4. 全连接层(由1-2个全连接层给出最后分类结果)
    5. Softmax层(主要用于分类)

## Image understanding with deep convolutional networks 使用深度卷积网络进行图像理解

#### 被使用的领域
the detection 检测, segmentation 分割 and recognition of objects 物体识别 and regions in images图像领域; face recognition人脸识别

algorithm parallelization并行计算


## Distributed representations and language processing 分布式知识表达和自然语言处理

#### 分布式知识表达比经典学习算法好

hidden layers 隐含层


## Recurrent neural networks RNN循环神经网络

#### 主要解决时序分析问题(语音识别，语言模型，机器翻译) 处理预测序列数据

state vector 状态向量 <br>
hidden units 隐含单元 <br>
Back-Propagation Through Time 沿时间反向传播<br>
English ‘encoder’ network 编码器网络<br>

#### long short-term memory LSTM长短时记忆网络
A special unit called the memory cell(记忆细胞) acts like an accumulator(累加器) or a gated leaky neuron(门控神经元): it has a connection to itself at the next time step(时间步长) that has a weight of one, so it copies its own real-valued state and accumulates the external signal, but this self-connection is multiplicatively gated by another unit that learns to decide when to clear the content of the memory