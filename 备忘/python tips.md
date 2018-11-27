## 类：

#### 类方法：

通过类来调用方法，而不是通过实例

`class_foo` 的参数是 cls，代表类本身，当我们使用 `A.class_foo()` 时，cls 就会接收 A 作为参数。另外，被 `classmethod` 装饰的方法由于持有 cls 参数，因此我们可以在方法里面调用类的属性、方法，比如 `cls.bar`

```python
class A(object):
    bar = 1
    @classmethod
    def class_foo(cls):
        print ('Hello, ', cls)
        print (cls.bar)

>>> A.class_foo()   # 直接通过类来调用方法
Hello,  <class '__main__.A'>
1
```

#### 静态方法：

在类中往往有一些方法跟类有关系，但是又不会改变类和实例状态的方法，这种方法是**静态方法**

静态方法没有 `self` 和 `cls` 参数

```python
class A(object):
    bar = 1
    @staticmethod
    def static_foo():
        print ('Hello, ', A.bar)

>>> a = A()
>>> a.static_foo()
Hello, 1
>>> A.static_foo()
Hello, 1
```

## glob方法：

glob可以用来查找文件，返回查找到的文件路径，可以用于加载数据集

```python
import glob
some_txt_dir='/home/txt/'
file_path=glob.glob(some_txt_dir+'*.txt')
# 将返回一个列表该目录下所包含所有.txt文件的路径列表
# 之后用相应的读取文件的方式来读取
```



## pytest 自动化测试：

- 测试文件以test\_开头（以_test结尾也可以）

- 测试类以Test开头，并且不能带有 \__init__ 方法

- 测试函数以test_开头

- 断言使用基本的assert

```python
py.test test_xx.py #可以这样用 
```


## skimage 用于python图像处理：

```python
#几乎集合了matlab的所有图像处理功能 io,transform,data等可以查API
import skimage
img=skimage.io.imread(path)#path是图片路径(fname,as_grey=True) as_grey是否读取灰度图片
skimage.io.imshow(img)#用来显示图像
skimage.io.imsave(path,img)#path储存路径
print(type(img))  #显示类型
print(img.shape)  #显示尺寸
print(img.shape[0])  #图片宽度
print(img.shape[1])  #图片高度
print(img.shape[2])  #图片通道数
print(img.size)   #显示总像素个数
print(img.max())  #最大像素值
print(img.min())  #最小像素值
print(img.mean()) #像素平均值
```

```python
from skimage import io, data, color,transform
import numpy as np

img = io.imread("./IMG-0003-00130.jpg")
gray = color.rgb2gray(img)

'''
测试一下图像旋转等操作
'''
rows,cols=gray.shape
print(gray.shape)
## 去除两边的操作：
gray[:,[x for x in range(rows) if x < 100 or x>400]]=0
## 图形旋转的操作，旋转15度但是不变
img1=transform.rotate(img,15)
#img1[:,[x for x in range(rows) if x < 100 or x>400]]=0
#img2=img1[[x for x in range(rows) if x > 100 and x<400],:]
img3=img1[100:400,100:400]
# img4=img1[100:400][:,100:400]
print(img3[139][206],img3[100][40])
print(type(img3[100:300][3]),type(img3[100][200]))
io.imshow(img3)
io.show()

##################################################################################


测试了一下label2rgb，大致范围为0-1，分为三个标签，变成三个颜色
print(gray[255,255])
rows, cols = gray.shape
labels = np.zeros([rows, cols])
for i in range(rows):
    for j in range(cols):
        if (gray[i, j] < 0.20):
            labels[i, j] = 0
        elif(gray[i,j]>0.8):
            labels[i, j] = 1
        else:
            labels[i,j]=2
####### 这个函数可以用于分割后对应标签转换为分割mask图
dst = color.label2rgb(labels)

io.imshow(dst)
io.show()
```

```python
import cv2
import numpy as np
from skimage import io, color

'''
这部分是opencv的东西
img = cv2.imread("./IMG-0003-00130.jpg", 0)
print(img.shape)
fi=img/255.0
gamma=0.2
out=np.power(fi,gamma)
img_blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
ret, img_th = cv2.threshold(img_blur, 100, 255, 0)  # 二值化
# cv2.imwrite("canny.jpg", cv2.Canny(img_th, 512, 512))
# cv2.imshow("A", cv2.imread("./canny.jpg"))
cv2.imshow("B", img_th)
cv2.imshow("gama",out)
cv2.waitKey()
cv2.destroyAllWindows()
'''
# 要用skimage可能要先安装matplotlib之类的
print(io.find_available_plugins())


# 读取灰度图片
def convert_gray(f):
    rgb = io.imread(f)
    return color.rgb2gray(rgb)


# 同时读取多个图片
paths = "./*.jpg"
coll = io.ImageCollection(paths, load_func=convert_gray)
# load_func默认是io.imread()，如果要读入灰度图，如例子中那样自己写一个函数
print(type(coll[0]))  # 得到的是ndarray 三个维度
io.imshow(coll[0])  # 需要安装matplotlib或者pil作为底层插件
io.show()
```

[skimage图像处理](https://www.jianshu.com/p/f2e88197e81d)

## Inspect：

inspect模块主要提供了四种用处：

1. 对是否是模块，框架，函数等进行类型检查。

2. 获取源码

3. 获取类或函数的参数的信息

4. 解析堆栈

## Assert 断言：

指的是程序进行到某个时间点，断定其必然是某种状态。用来检查一个条件，如果它为真，就不做任何事。如果它为假，则会抛出AssertError并且包含错误信息

assert \<test>,\<message>

VGG16中用法：

```python
assert red.get_shape().as_list()[1:] == [224, 224, 1]
#判断是否是224,224,1 长，宽，通道，因为VGG的输入为224*224的图像，如果不是的话报异常
```

## Numpy

`np.mgrid[0:5,0:3]`的意思是相当于 画一个网格一样的点,横坐标从0开始画5个长度,纵坐标从0开始画3个

返回两个dim `nodes=np.mgrid[0:5,0:3]`，那么`nodes[0]`即所有点横坐标，`nodes[1]`即所有点的纵坐标

reference : [mgrid](https://drivingc.com/numpy/5af5976b2392ec315d3ccecc)

```python
# 最后得到的：
array([[[0,0,0],
        [1,1,1],
        [2,2,2],
        [3,3,3],
        [4,4,4]],
        
       [[0,1,2],
        [0,1,2],
        [0,1,2],
        [0,1,2],
        [0,1,2]]])
```

## matplotlib

`%matplotlib inline` 在`jupyter notebook`中很常见，具体来说可以在`Ipython`编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉`plt.show()`这一步

