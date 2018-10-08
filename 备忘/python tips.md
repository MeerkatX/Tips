## pytest 自动化测试

- 测试文件以test\_开头（以_test结尾也可以）

- 测试类以Test开头，并且不能带有 \__init__ 方法

- 测试函数以test_开头

- 断言使用基本的assert

```python
py.test test_xx.py #可以这样用 
```


## skimage 用于python图像处理

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

## Inspect

inspect模块主要提供了四种用处：

1. 对是否是模块，框架，函数等进行类型检查。

2. 获取源码

3. 获取类或函数的参数的信息

4. 解析堆栈

## Assert 断言

指的是程序进行到某个时间点，断定其必然是某种状态。用来检查一个条件，如果它为真，就不做任何事。如果它为假，则会抛出AssertError并且包含错误信息

assert \<test>,\<message>

VGG16中用法：

```python
assert red.get_shape().as_list()[1:] == [224, 224, 1]
#判断是否是224,224,1 长，宽，通道，因为VGG的输入为224*224的图像，如果不是的话报异常
```