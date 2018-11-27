# pydicom用法备忘

## 官方文档简介：

pydicom is a pure python package for working with [DICOM](http://medical.nema.org/) files. It was made for inspecting and modifying DICOM data in an easy "pythonic" way. The modifications can be written again to a new file.

As a pure python package, pydicom can run anywhere python runs without any other requirements, although [NumPy](http://www.numpy.org/) is needed if manipulating pixel data.

pydicom is not a DICOM server, and is not primarily about viewing images. It is designed to let you manipulate data elements in DICOM files with python code.

Limitations -- for files with *compressed* pixel data, pydicom can decompress it (with additional libraries installed) and allow you to manipulate the data, but can only store changed pixel data as uncompressed. Files can always be read and saved (including compressed pixel data that has not been modified), but once decompressed, modified pixel data cannot be compressed again.

## 常用方法：

```python
import pydicom as dicom

## 读取dicom文件，根据dcm_dir即dcm文件位置：
dcm=dicom.read_file(dcm_dir)
## 即将piexl转化为ndarray，可以用numpy来进行接下来的处理，貌似是私有变量，不可直接赋值
## 即 dcm.piexl_array= dcm.pixel_array * mask 会报错
dcm.piexl_array
## 将其转化为HU值的图像，WL_extract，WW_extract ，窗宽：
img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
# 拿到 提取轮廓的窗，并规范化
img_extract = (img_origin - (WL_extract - WW_extract / 2)) / WW_extract * 255  # 规范化到0-255

# 对像素修改后再保存为dcm，假设 dcm*mask 这个是numpy的两矩阵相乘，需要转化为bytes后赋值给PixelData
dcm.PixelData = (dcm.pixel_array * mask).tobytes()
# 保存 保存的位置和文件名'.dcm'
dcm.save_as(numpy_2_dcm_dir + 'IMG-0001-{0}.dcm'.format(i + 1))
```

## Core elements in pydiocm:

Dataset是`dicom.read_file(dcm_dir)`的返回类型，定义了很多对于我来说不重要的属性，比如用户姓名，拍摄时间等信息。

## pixel Data:

#### PixelData:

`PixelData` contains the raw bytes exactly as found in the file. If the image is JPEG compressed, these bytes will be the compressed pixel data, not the expanded, uncompressed image. Whether the image is e.g. 16-bit or 8-bit, multiple frames or not, `PixelData`contains the same raw bytes. But there is a function that can shape the pixels more sensibly if you need to work with them …

#### pixel_array:

A property of [`dataset.Dataset`](https://pydicom.github.io/pydicom/stable/api_ref.html#pydicom.dataset.Dataset) called `pixel_array` provides more useful pixel data for uncompressed and compressed images ([decompressing compressed images if supported](https://pydicom.github.io/pydicom/stable/image_data_handlers.html)). The `pixel_array` property returns a NumPy array:

```python
>>> ds.pixel_array 
array([[ 905, 1019, 1227, ...,  302,  304,  328],
       [ 628,  770,  907, ...,  298,  331,  355],
       [ 498,  566,  706, ...,  280,  285,  320],
       ...,
       [ 334,  400,  431, ..., 1094, 1068, 1083],
       [ 339,  377,  413, ..., 1318, 1346, 1336],
       [ 378,  374,  422, ..., 1369, 1129,  862]], dtype=int16)
>>> ds.pixel_array.shape
(64, 64)
```

上面保存的示例代码：

```python
for n,val in enumerate(ds.pixel_array.flat): # example: zero anything < 300
    if val < 300:
        ds.pixel_array.flat[n]=0
ds.PixelData = ds.pixel_array.tobytes()
ds.save_as("newfilename.dcm")

import os
os.remove("newfilename.dcm")
```

其余可以参考文档

## Reference

[Pydicom User Guide](https://pydicom.github.io/pydicom/stable/pydicom_user_guide.html)



