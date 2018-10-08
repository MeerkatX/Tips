## tf.argmax(input,axis=None,name=None,dimension=None,output_type=tf.int64)

```python
import tensorflow as tf
import numpy as np

A = [[1, 3, 4, 5, 6]]
B = [[1, 3, 4], [2, 4, 1]]

with tf.Session() as sess:
    print(sess.run(tf.argmax(A, 0))) #输出[0,0,0,0,0]因为axis=0
    print(sess.run(tf.argmax(B, 1))) #输出[2,1]因为axis=1，输出最大值所在位置[1,3,4] [2,4,1]
    							  #                                      0 1 2   0 1 2
```

