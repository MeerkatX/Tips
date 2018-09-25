# VGG
## Training
#### 参数设置
- using mini-batch gradient descent with momentum.The batch size was set to **256**,momentum to **0.9**. 随机（批量）梯度下降+动量

- L2正则以及dropout. The training was regularised by weight decay (the L2 penalty multiplier set to $5^{10^{-4}}$) and dropout regularisation for the first two fully-connected layers (dropout ratio set to **0.5**).

- 学习率设置，当准确度不再提升时减少学习率 The learning rate was initially set to $10^{−2}$. decreased by a factor of 10 when the validation set accuracy stopped improving
#### 参数初始化
- 先训练一个浅的模型 began with training the configuration A, shallow enough to be trained with random initialisation

- 当训练更深的模型的时候 we initialised the first four convolutional layers and the last three fullyconnected layers with the layers of net A (the intermediate layers were initialised randomly).

- 并且在这个训练中不减少学习率 We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning.
  $$
  
  $$

