# 消融实验

数据集：

1. [Safety Helmet Detection(kaggle.com)](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
2. [Hard Hat Workers Dataset (roboflow.com)](https://public.roboflow.com/object-detection/hard-hat-workers)



train：4500

val：500

## 基础参数设置

```
default=8		 1 epochs completed in 0.017 hours.
default=32	 1 epochs completed in 0.013 hours.
default=64   1 epochs completed in 0.012 hours.
default=128  1 epochs completed in 0.014 hours.
```

batch-size: 32

Image-size: 640

```python
# Parameters
nc: 2  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.25  # layer channel multiple
```



## baseline

- exp25

- no-pretrain

  <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220408230521207.png" alt="image-20220408230521207" style="zoom:50%;" />



# 预训练和batch测试

- exp11

  cfg:yolov5n.yaml

  Model size:3.9M

  Pretrain: yolov5n

  batch-size: 32

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407093516688.png" alt="image-20220407093516688" width="800" />

- exp15

  cfg：yolov5n.yaml

  Model size:3.9M

  Pretrain：none
  batch-size：128

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407093558329.png" alt="image-20220407093558329" width="800"/>



##  batch测试

- exp28
- yolov5n-Helmet.yaml
- batch：128

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204092125408.png" alt="image-20220409212506365" style="zoom:50%;" />



# 轻量化网络

## Shufflenetv2

> [Cite]Ma, Ningning, et al. “Shufflenet v2: Practical guidelines for efficient cnn architecture design.” Proceedings of the European conference on computer vision (ECCV). 2018.
>
> [论文地址](https://arxiv.org/abs/1807.11164)
>
> [论文代码](https://github.com/megvii-model/ShuffleNet-Series)

v1主要用的分组卷积



### 实验结果

- exp17

  cfg: yolov5-shufflenetv2.yaml

  Model size: 805kB

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407093620656.png" alt="image-20220407093620656" width="800" />



## Mobilenetv3

> [Cite]Howard, Andrew, et al. “Searching for mobilenetv3.” Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
>
> [论文地址](https://arxiv.org/abs/1905.02244)
>
> [论文代码](https://github.com/xiaolai-sqlai/mobilenetv3)

### overview

深度可分离卷积(depth-wise convolution)

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091548744.png" alt="image-20220409154846720" width="500" />

1. 更新Block（bneck）

2. 使用NAS搜索参数
3. 重新设计耗时层结构

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091549079.png" alt="image-20220409154938056" width="500" />

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091550226.png" alt="image-20220409155041189" width="500" />



#### v3相比v2

- 更新Block

1. 加入SE模块

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204100052835.png" alt="SE" style="zoom:50%;" />

2. 更新激活函数

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091556077.png" alt="image-20220409155605016" width="500" />

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091557420.png" alt="image-20220409155718393" width="500" />

### 实验结果

- exp18
- cfg: yolov5-mobilenetv3-small.yaml
- Model size: 2.0M

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407124323895.png" alt="image-20220407124323895"  width="800" />



## Ghostnet

> Han, Kai, et al. “Ghostnet: More features from cheap operations.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
>
> [论文地址](https://arxiv.org/abs/1911.11907)
>
> [论文代码](https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch)





### 实验结果

#### test1: v6.0

- cfg: yolov5n-ghost.yaml

- exp19
- Model size: 2.4M

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407161248022.png" alt="image-20220407161248022" width="800" />

#### test2: v6.1

- cfg: yolov5n-ghost-v61.yaml

- exp20

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407195212431.png" alt="image-20220407195212431" width="800" />

#### 网络对比

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407144724940.png" alt="image-20220407144724940" width="1000" />



##### focus->conv

link: https://github.com/ultralytics/yolov5/issues/4825#issue-998038464

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204100049016.png" alt="Focus" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204100049490.png" alt="Conv" style="zoom:50%;" />

# 网络优化

## ACON激活函数

> Ma, Ningning, et al. “Activate or not: Learning customized activation.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
>
> [论文地址](https://arxiv.org/abs/2009.04759)
>
> [论文代码](https://github.com/nmaac/acon/blob/main/acon.py)





## 注意力机制

### CBAM

> Woo, Sanghyun, et al. “Cbam: Convolutional block attention module.” *Proceedings of the European conference on computer vision (ECCV)*. 2018.
>
> [论文地址](https://arxiv.org/abs/1807.06521)

#### overview

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091659101.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091700478.png" alt="img" style="zoom:50%;" />

#### 实验结果

- exp21
- cfg: yolov5-cbam.yaml

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407232545101.png" alt="image-20220407232545101" width="800" />



### CA

> Hou, Qibin, Daquan Zhou, and Jiashi Feng. “Coordinate attention for efficient mobile network design.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
>
> [论文地址](https://arxiv.org/abs/2103.02907)
>
> [论文代码](https://github.com/Andrew-Qibin/CoordAttention)

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204100047262.png" alt="CA" style="zoom:50%;" />

#### 实验结果

- exp22
- yolov5-ca.yaml

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220408115306886.png" alt="image-20220408115306886" width="800" />



## BiFPN特征融合

>[Cite]Tan, Mingxing, Ruoming Pang, and Quoc V. Le. “Efficientdet: Scalable and efficient object detection.” Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
>
>[论文地址](https://arxiv.org/abs/1911.09070)
>
>[论文代码](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)





### 实验结果

- exp23
- cfg: yolov5-bifpn.yaml

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220408151444731.png" alt="image-20220408151444731" width="800" />





## Transfomer

> [论文地址](https://arxiv.org/abs/2010.11929)
>
> [论文代码]()

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204100049079.png" alt="Transformer" style="zoom:50%;" />

- **exp26**
- cfg：yolov5n-transformer.yaml

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091238744.png" alt="image-20220409123824716" width="800" />



<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091438730.png" alt="image-20220409143834670" width="800" />





## swinTransformer

>[论文连接]
>
>[代码连接]



## YOLOv5+Ghostconv+BiFPN+CA



### 实验结果

- exp24
- cfg: yolov5-Ghostconv-BiFPN-CA

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220408185802294.png" alt="image-20220408185802294" width="800" />







# Results

| train/exp | Model                      | size     (pixels) | mAPval     0.5:0.95 | mAPval     0.5 | FLOPs-train | params(M)   | FLOPs     @640 (B) | val/exp | Speed     V100 b32     (ms) | detect-inference-1  (ms) | Speed     jetson nano     (ms) |
| --------- | -------------------------- | ----------------- | ------------------- | -------------- | ----------- | ----------- | ------------------ | ------- | --------------------------- | ------------------------ | ------------------------------ |
| 25        | yolov5n                    | 640               | 0.607               | 0.937          | 3.9         | 1.76        | 4.5                | 1       | 1.4                         | 9.4                      |                                |
| 17        | yolov5n-shufflenetv2       | 640               | 0.513               | 0.865          | 0.5         | 0.22        | 0.5                | 2       | 0.5                         | 10.6                     |                                |
| 18        | yolov5n-mobilenetv3        | 640               | 0.564               | 0.91           | 1.2         | 0.79        | 1.2                | 3       | 1.1                         | 14.5                     |                                |
| 19        | yolov5n-ghost              | 640               | 0.599               | 0.934          | 2.3         | 0.94        | 2.3                | 4       | 1..0                        | 13.4                     |                                |
| 20        | yolov5n-ghost-v61          | 640               | 0.592               | 0.933          | 2.3         | 0.94(0.939) | 2.3                | 5       | 1.4                         | 13.9                     |                                |
| 21        | yolov5n-cbam               | 640               | 0.61                | 0.939          | 3.8         | 1.69        | 4.1                | 6       | 1.9                         | 15.3                     |                                |
| 22        | yolov5n-ca                 | 640               | 0.618               | 0.946          |             | 1.77        |                    | 7       | 1.6                         | 10.9                     |                                |
| 23        | yolov5n-bifpn              | 640               | 0.612               | 0.939          | 4.2         | 1.78        | 4.2                | 8       | 1.4                         | 0.9                      |                                |
| 24        | yolov5n-Ghostconv-BiFPN-CA | 640               | 0.606               | 0.94           |             | 1.49        |                    | 9       | 1.5                         | 10.2                     |                                |
|           |                            |                   |                     |                |             |             |                    |         |                             |                          |                                |



## 对比：

yolov5n误判：

005308、005316、005322、005354、005395

特殊情况无法判断：005381(蹲下只有头盔)

小目标漏检：005351

复杂情况：005384



| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/005308_jpg.rf.4403a2f5f05f7715de6bdd234a24f8e3.jpg" alt="005308_jpg.rf.4403a2f5f05f7715de6bdd234a24f8e3" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090131552.jpg" alt="005308_jpg.rf.4403a2f5f05f7715de6bdd234a24f8e3" style="zoom:50%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/005316_jpg.rf.45c8dd29cadc2f60f46a4ce0a4f9b9b9.jpg" alt="005316_jpg.rf.45c8dd29cadc2f60f46a4ce0a4f9b9b9" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090143364.jpg" alt="005316_jpg.rf.45c8dd29cadc2f60f46a4ce0a4f9b9b9" style="zoom:50%;" /> |
| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/005322_jpg.rf.38c87c9a9999606bd5936d6ddb6915b2.jpg" alt="005322_jpg.rf.38c87c9a9999606bd5936d6ddb6915b2" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090132797.jpg" alt="005322_jpg.rf.38c87c9a9999606bd5936d6ddb6915b2" style="zoom:50%;" /> |
| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/005354_jpg.rf.66ac8412f01e08a6146d5bc63bbb158c.jpg" alt="005354_jpg.rf.66ac8412f01e08a6146d5bc63bbb158c" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090133114.jpg" alt="005354_jpg.rf.66ac8412f01e08a6146d5bc63bbb158c" style="zoom:50%;" /> |
| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/005395_jpg.rf.a38830389c0d26e139ba81b2e1a813a3.jpg" alt="005395_jpg.rf.a38830389c0d26e139ba81b2e1a813a3" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090136681.jpg" alt="005395_jpg.rf.a38830389c0d26e139ba81b2e1a813a3" style="zoom:50%;" /> |
|                                                              |                                                              |



| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090137571.jpg" alt="005381_jpg.rf.be7a0d23f9a13d1e0321014db768002f" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090137290.jpg" alt="005381_jpg.rf.be7a0d23f9a13d1e0321014db768002f" style="zoom:50%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                                                              |                                                              |

| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090138086.jpg" alt="005351_jpg.rf.ed2d454b2e059fc13f901bcc62947bfa" style="zoom:50%;" /> | :x:  |
| :----------------------------------------------------------: | :--: |
| <img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204090139807.jpg" alt="005384_jpg.rf.34ab47f3859c29a33df2780579333dc1" style="zoom:50%;" /> | :x:  |
|                                                              |      |





# 其他测试

## SPP-SPPF

`SPP`:将输入并行通过多个不同大小的`MaxPool`，然后做进一步融合，能在一定程度上解决目标多尺度问题。

而`SPPF`结构是将输入串行通过多个`5x5`大小的`MaxPool`层，这里需要注意的是串行两个`5x5`大小的`MaxPool`层是和一个`9x9`大小的`MaxPool`层计算结果是一样的，串行三个`5x5`大小的`MaxPool`层是和一个`13x13`大小的`MaxPool`层计算结果是一样的。



<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204100048248.png" alt="SPP" style="zoom:50%;" />

**SPP vs SPPF**

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091152646.png" alt="image-20220409115240561" style="zoom:50%;" />





## yolov5n6

- **exp27**
- cfg：yolov5n6

![image-20220409161148223](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204091611250.png)





## tph-yolov5测试

>[论文地址]()
>
>[论文代码]()





## YOLOX

> [论文地址]([[2107.08430\] YOLOX: Exceeding YOLO Series in 2021 (arxiv.org)](https://arxiv.org/abs/2107.08430))
>
> [论文代码]([Megvii-BaseDetection/YOLOX: YOLOX is a high-performance anchor-free YOLO, exceeding yolov3~v5 with MegEngine, ONNX, TensorRT, ncnn, and OpenVINO supported. Documentation: https://yolox.readthedocs.io/ (github.com)](https://github.com/Megvii-BaseDetection/YOLOX))





# 调参

## lrf(最终 OneCycleLR 学习率)

[CosineAnnealingLR和OneCycleLR的原理与使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/350712244)



### lrf: 0.01

- exp28
- yolov5n-Helmet.yaml
- batch：128
- data/hyps/hyp.scratch-low.yaml
- lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf)

**[result](https://wandb.ai/yin-qiyu/helmet/runs/10g70rjo?workspace=user-yin-qiyu)**





### lrf: 0.1

- exp29

- yolov5n-Helmet.yaml
- batch：128
- data/hyps/hyp.scratch-low.yaml
- lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)

**[result](https://wandb.ai/yin-qiyu/helmet/runs/j01nvbgy?workspace=user-yin-qiyu)**









# TO DO

+ [x] backbone: ShuffleNetV2
+ [x] backbone: Mobilenetv3
+ [x] backbone: Ghostnet:star:
+ [x] backbone：CBAN
+ [x] backbone：CA
+ [x] head:BiFPN
+ [x] YOLOv5+Ghostconv+BiFPN+CA⭐️
+ [x] backbone: c3tr
+ [ ] Backbone:c3str
+ [ ] Prune: FSP
+ [ ] contrast🚀
+ [ ] tph-yolov5
+ [ ] yolox
+ [ ] hyp
  + [ ] lrf






