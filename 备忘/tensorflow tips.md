#### `tf.concat`用法

可以链接两个矩阵

`tf.concat(concat_dim, values, name='your name')`要求的参数

`concat_dim`表明在哪个维度上链接，`values`表明要链接的两个tensor

#### `tf.argmax`的用法

与`numpy.argmax`效果相似

```python
import tensorflow as tf
import numpy as np

A = [[1, 3, 4, 5, 6]]
B = [[1, 3, 4], [2, 4, 1]]

with tf.Session() as sess:
    #tf.argmax(input,axis=None,name=None,dimension=None,output_type=tf.int64)
    print(sess.run(tf.argmax(A, 0))) #输出[0,0,0,0,0]因为axis=0
    print(sess.run(tf.argmax(B, 1))) #输出[2,1]因为axis=1，输出最大值所在位置[1,3,4] [2,4,1]
    							  #                                      0 1 2   0 1 2
```

#### `tf.name_scope` 和 `tf.variable_scope` 

`tf.variable_scope`可以让变量有相同的命名，包括`tf.get_variable`得到的变量，还有`tf.Variable`的变量

`tf.name_scope`可以让变量有相同的命名，只是限于`tf.Variable`的变量

#### `tf.get_variable`和`tf.Variable`

`tf.Variable`用于生成一个初始值为`initial-value`的变量**必须指定初始化值**

`tf.get_variable`获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。 
**可以用各种初始化方法，不用明确指定值。**

#### 参数为 `-1` 的情况

对于普通列表来说-1是最后一个元素

对于`numpy`来说`np.reshape(z,[-1,1])`是会根据另一个参数维度计算出数组的shape，如例子中[-1,1]来说，会变为一列，自动计算有几行 

