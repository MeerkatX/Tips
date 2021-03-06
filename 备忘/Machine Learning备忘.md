# Cross Entropy

### 香农 信息熵：

[如何通俗的理解交叉熵和相对熵](https://www.zhihu.com/question/41252833)

引用其中猜球的游戏，当对整个系统一无所知时，假设所有概率相等即**最大熵原理**。

![imgs](https://pic2.zhimg.com/80/v2-97e76bd3402b6d765bfc1934d4c75f75_hd.png)

这时候，假设$\frac{1}{4}$ 是每个球的概率，需要猜两次才能得出结果：

那么$H=\frac{1}{4}\times 2 + \frac{1}{4}\times 2+\frac{1}{4}\times 2+\frac{1}{4}\times 2=2$ 

**信息熵代表的是随机变量或整个系统的不确定性，熵越大，随机变量或系统的不确定性就越大** 如上，信息熵最大，因为其一无所知

总结规律，香农提出了针对概率为$p$的小球，需要**猜的次数**为$\log_2\frac{1}{p}$，之后对其求**期望**(均值)：
$$
\sum^N_{k=1}p_k\log_2\frac{1}{p_k}
$$
这就是信息熵。

### 交叉熵

**用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小**。即其真实的分布为$\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{8}$，但是由于对其一无所知，假设是四个$\frac{1}{4}$ 

那么得到的交叉熵是

$H=\frac{1}{2}\times2+\frac{1}{4}\times2+\frac{1}{8}\times2+\frac{1}{8}\times2=2$

交叉熵的公式为：
$$
\sum^N_{k=1}p_k\log_2\frac{1}{q_k}
$$
$p_k$真实分布，$q_k$非真实分布

我们总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明我们算法所算出的非真实分布越接近真实分布。

#### 交叉熵为什么用在机器学习中：

[交叉熵代价函数](https://blog.csdn.net/u012162613/article/details/44239919)

因为sigmoid函数的性质，导致σ′(z)在z取大部分值时会很小（两端，几近于平坦），这样会使得w和b更新非常慢。为了克服这个缺点，引入了交叉熵代价函数
$$
C=-\frac{1}{n}\sum_x[y\ln a+(1-y)\ln(1-a)]
$$

$$
a=σ(z),z=\sum W_j*x_j+b
$$

对其求导，消除了σ′(z)
$$
\frac{\partial C}{\partial w_j}=\frac{1}{n}\sum_x x_j(\sigma(z)-y)
$$

$$
\frac{\partial C}{\partial b}=\frac{1}{n}\sum_x x_j(\sigma(z)-y)
$$

导数中没有σ′(z)这一项，权重的更新是受σ(z)−y这一项影响，即受误差的影响。所以当误差大的时候，权重更新就快，当误差小的时候，权重的更新就慢。这是一个很好的性质。

### KL散度 相对熵

**相对熵，其用来衡量两个取值为正的函数或概率分布之间的差异** 相对熵 = 某个策略的交叉熵 - 信息熵

$KL(p||q)=H(p,q)-H(p)$
$$
KL=\sum^N_{k=1}p_k\log_2\frac{p_k}{q_k}
$$


