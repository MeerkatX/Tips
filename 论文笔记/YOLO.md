# You Only Look Once: Uniﬁed(统一的),Real-Time Object Detection

YOLO将目标识别问题看作是一个回归问题：**利用整张图作为网络的输入，直接在图像的多个位置上回归出这个位置的目标边框，以及目标所属的类别。**区别于RCNN系列，当YOLO出来的时候已经有了fast RCNN。YOLO的实现比较简单，快速，但是相对的$mAP$较RCNN低，以及不能定位很准，识别小目标(因为只在最后一层进行目标的检测回归)

1. 给个一个输入图像，首先将图像划分成7\*7的网格
2. 对于每个网格，我们都预测2个边框（包括每个边框是目标的置信度以及每个边框区域在多个类别上的概率）
3. 根据上一步可以预测出7\*7\*2个目标窗口，然后根据阈值去除可能性比较低的目标窗口，最后NMS去除冗余窗口即可

## Abstract

  we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. 目标识别看作一个(空间bb和类的可能性)回归问题。实现了end-to-end

## Unified Detection

Our system divides the input image into an $S\times S$ grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.将输入图像分为$S \times S$的网格，然后看要识别的物体是否在这个网格的中心，如果在中心，则这个网格负责检测这个物体。

Each grid cell predicts B bounding boxes and conﬁdence scores for those boxes. 每个网格预测**B**个BB以及这些BB中分类的分数(置信度)

## 预训练

For pretraining we use the ﬁrst 20 convolutional layers from Figure3 followed by a average-pooling layer and a fully connected layer. 

## 输入

预训练20层卷积层时是半分辨率，训练的时候$448\times448$作为输入图像（因为检测需要细粒度的视觉信息）并且新添加了4个卷积层和2个全连接层

## grid cell

It divides the image into an S×S grid and for each grid cell predicts B bounding boxes, conﬁdence for those boxes, and C class probabilities. These predictions are encoded as an $S \times S \times (B \times 5 + C)$tensor. 对于 YOLO来说，是分成$7\times7$ 每个grid cell中预测 **2** 个BB 每个BB 有 **5** 个参数即 **(x,y,w,h,confidence)** 以及**C**个种类 也就是 $7\times7\times(2\times5+20)$ 最后得到1470个数字

## 一些参数

batch size = 64 , momentum = 0.9 , decay = 0.0005 , learning rate = $10^{-3}$ to $10^{-2}$ 以及训练$10^{-4}$ 同时训练使用了**dropout** ( dropout = .5,after the first connected layer ) 以及**extensive data augmentation**（random scaling translations of up to $20$）

## 激活函数

leaky rectified linear activation：
$$
\phi(n)=
\begin{cases}
x,& \text{if $x$ > 0}\\[2ex]
0.1x,& \text{otherwise}
\end{cases}
$$


## LOSS函数

这里面的$\mathbb{1}_{ij}^{obj}$的意思是如果目标在这第 $i$ 个cell并且是第 $j$ 个bb的预测的话就为1，否则为0

其中$\lambda_{coord}=5$和$\lambda_{noobj}=.5$
$$
\lambda_{coord}\sum^{S^2}_{i=0}\sum^B_{j=0}\mathbb{1}^{obj}_{ij}\big[(x_i-\hat x_i)^2+(y_i-\hat y_i)^2\big] \\+\lambda_{coord}\sum^{S^2}_{i=0}\sum^B_{j=0}\mathbb{1}^{obj}_{ij}\big[(\sqrt{w_i}-\sqrt{\hat w_i})^2+\big(\sqrt h_i-\sqrt{\hat h_i}\big)^2\big]
\\+\sum^{S^2}_{i=0}\sum^B_{j=0}\mathbb{1}^{obj}_{ij}\big(C_i-\hat{C_i}\big)^2
\\+\lambda_{noobj}\sum^{S^2}_{i=0}\sum^B_{j=0}\mathbb{1}^{noobj}_{ij}\big(C_i-\hat{C_i}\big)^2
\\+\sum^{S^2}_{i=0}\mathbb{1}^{obj}_i\sum_{c\in{classes}}(p_i(c)-\hat p_i(c))^2
$$

## 代码

输出解析的代码需要仔细看一下，暂时不理解：

```python
#解析输出
def interpret_output(self,output):
    	#这里是7*7*30个预测值
		probs = np.zeros((7,7,2,20))
         #前0-980为20类的分类的confidence将其变为7*7*20
		class_probs = np.reshape(output[0:980],(7,7,20))
         #之后的980-1078为两个B-Box的confidence将其变为7*7*2
		scales = np.reshape(output[980:1078],(7,7,2))
         #剩下的部分是每个bbox的(x,y,w,h)
         boxes = np.reshape(output[1078:],(7,7,2,4))
         #偏移，0-6 * 14 个 reshape 之后 为2 7 7 
         '''
         [[ [0-6]              之后的transpose将(0,1,2)的轴装置为(1,2,0)
            [0-6]              [ [ [0,0],[1,1],[2,2]...[6,6](共7个) ],
            ...                  [ [0,0],[1,1],[2,2]...[6,6] ],
            [0-6] ]
                                 ...
          [ [0-6]                [ [0,0],[1,1],[2,2]...[6,6] ](共7个)  ]
            [0-6]
            ...
            [0-6] ]
           ]
         '''
		offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

		boxes[:,:,:,0] += offset# 对x加偏移
		boxes[:,:,:,1] += np.transpose(offset,(1,0,2))# 对y加偏移
		boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0 # 求x,y的平均
		boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])# h的平方
		boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])# w的平方
		
		boxes[:,:,:,0] *= self.w_img
		boxes[:,:,:,1] *= self.h_img
		boxes[:,:,:,2] *= self.w_img
		boxes[:,:,:,3] *= self.h_img

		for i in range(2):
			for j in range(20):
				probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

		filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs)
		boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]
		
		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0 : continue
			for j in range(i+1,len(boxes_filtered)):
				if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
					probs_filtered[j] = 0.0
		
		filter_iou = np.array(probs_filtered>0.0,dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]

		result = []
		for i in range(len(boxes_filtered)):
			result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

		return result

```



## Reference

[非极大值抑制NMS](https://www.julyedu.com/question/big/kp_id/26/ques_id/2141)

[YOLO-TensorFlow](https://github.com/gliese581gg/YOLO_tensorflow)