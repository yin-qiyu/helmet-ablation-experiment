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

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204092125408.png" alt="image-20220409212506365" width="800" />



# 轻量化网络

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204141946256.png" alt="轻量化神经网络" width="500" />



# 轻量化

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

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204170011394.png" alt="image-20220417001114306" style="zoom:50%;" />

1. 减少一个卷积层的卷积核个数（32->16）

2. 精简Last stage

![image-20220417001258583](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204170012613.png)



重新设计激活函数

![image-20220417001420419](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204170014451.png)

![image-20220417001554740](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204170015780.png)









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



## EfficientNetV2

>[论文](https://arxiv.org/abs/2104.00298)





# 网络优化

## Focus换conv

[Is the Focus layer equivalent to a simple Conv layer? · Issue #4825 · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/issues/4825)

```python
YOLOv5 🚀 2022-3-15 torch 1.11.0+cu102 CUDA:0 (Tesla V100-PCIE-32GB, 32510.5MB)

      Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
        7040       23.07         2.259         8.273         19.77       (16, 3, 640, 640)      (16, 64, 320, 320)
        7040       23.07         1.860         28.66         38.16       (16, 3, 640, 640)      (16, 64, 320, 320)
        7040       23.07         1.919         8.144         19.86       (16, 3, 640, 640)      (16, 64, 320, 320)
        7040       23.07         1.860         27.21         39.12       (16, 3, 640, 640)      (16, 64, 320, 320)

Process finished with exit code 0
```

![image-20220419205619926](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204192056965.png)



## spp换sppf

![image-20220419205726284](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204192057344.png)

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



```python
# ---------------------------- CBAM start ---------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)
class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out

```



#### 实验结果

- exp21
- cfg: yolov5-cbam.yaml

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407232545101.png" alt="image-20220407232545101" width="800" />



### CA（位置注意力）

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





## SE（挤压-激励注意力）

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202205031837551.png" alt="image-20220503183736450" style="zoom: 33%;" />

```python
class SE(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(SE, self).__init__()
        #c*1*1 c:channel number
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
```





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





## PP-LCNet-1x



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

复杂情况（像素过低or情况复杂）：005384



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



## 数据增强



<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204171147435.jpg" alt="preview" style="zoom:50%;" />

总共分成上述 4 个步骤，整体流程如下图所示(前两步是马赛克增强，第三步是几何变换增强，第 4 步是 MixUp 增强

\1) 马赛克增强

1. 随机出 4 张图片在待输出图片中交接的中心点坐标
2. 随机出另外 3 张图片的索引以及读取对应的标注
3. 对每张图片采用保持宽高比的 resize 操作缩放到指定大小
4. 按照上下左右规则，计算每张图片在待输出图片中应该放置的位置，因为图片可能出界故还需要计算裁剪坐标
5. 利用裁剪坐标将缩放后的图片裁剪，然后贴到前面计算出的位置，其余位置全部补 114 像素值
6. 对每张图片的标注也进行相应处理
7. 由于拼接了 4 张图，所以输出图片大小会扩大 4 倍

\2) 几何变换增强

random_perspective 包括平移、旋转、缩放、错切等增强，并且会将输入图片还原为 (640, 640)，同时对增强后的标注进行处理，过滤规则是

1. 增强后的 gt bbox 宽高要大于 wh_thr
2. 增强后的 gt bbox 面积和增强前的 gt bbox 面积要大于 ar_thr，防止增强太严重
3. 最大宽高比要小于 area_thr，防止宽高比改变太多

\3) MixUp

Mixup 实现方法有多种，常见的做法是：要么 label 直接拼接起来，要么 label 也采用 alpha 混合，作者的做法非常简单，对 label 直接拼接即可，而图片也是采用固定的 0.5:0.5 混合方法。

其处理流程是：

1. 随机出一张图片，必须要保证该图片不是空标注
2. 对随机出的图片采用保持宽高比的 resize 操作缩放到指定大小
3. 然后左上 padding 成指定大小，padding 值也是 114
4. 对 padding 后的图片进行随机抖动增强
5. 随机采用 flip 增强
6. 如果处理后的图片比原图大，则还需要进行随机裁剪增强
7. 对标签进行对应处理，并且采用和马赛克增强一样的过滤规则
8. 如果过滤后还存在 gt bbox，则采用 0.5:0.5 的比例混合原图和处理后的图片，标签则直接拼接即可

\4) 图片后处理

图片后处理操作也包括众多数据增强操作，如下所示：

1. 随机 ColorJit，包括众多颜色相关增强
2. 随机翻转增强
3. 对随机后的图片采用保持宽高比的 resize 操作缩放到指定大小
4. 对于宽高小于 8 像素的 gt bbox 直接删掉，因为网络输出的最小 stride 是 8
5. Padding 成正方形图片输出

 

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





# 超参

网络搜索，随机搜索，贝叶斯搜索

ssh -p 17025 root@region-11.autodl.com 

ssh -p 24454 root@region-11.autodl.com 

wandb login --relogin

![image-20220410135410006](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204101354035.png)





## 遗传算法

**exp30**

exp31





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



# 待解决

## 项目落地

- [ ] 小目标数据集

- [ ] 



- 简单数据集：
  - 特征明显
  - 类别少
  - 目标大
