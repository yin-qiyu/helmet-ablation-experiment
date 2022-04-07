# 轻量化网络-消融实验

数据集：[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)

train：4500

val：500

## 基础参数设置

batch-size: 32

Image-size: 640

```python
# Parameters
nc: 2  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.25  # layer channel multiple
```

## 预训练测试

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



## Shufflenetv2

> [Cite]Ma, Ningning, et al. “Shufflenet v2: Practical guidelines for efficient cnn architecture design.” Proceedings of the European conference on computer vision (ECCV). 2018.
>
> [论文地址](https://arxiv.org/abs/1807.11164)
>
> [论文代码](https://github.com/megvii-model/ShuffleNet-Series)

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

### test1: v6.0

- cfg: yolov5n-ghost.yaml

- exp19
- Model size: 2.4M

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407161248022.png" alt="image-20220407161248022" width="800" />

### test2: v6.1

- cfg: yolov5n-ghost-v61.yaml

- exp20

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407195212431.png" alt="image-20220407195212431" width="800" />

### 网络对比

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407144724940.png" alt="image-20220407144724940" width="1000" />



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

- exp21
- cfg: yolov5-cbam.yaml





### CA

> Hou, Qibin, Daquan Zhou, and Jiashi Feng. “Coordinate attention for efficient mobile network design.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
>
> [论文地址](https://arxiv.org/abs/2103.02907)
>
> [论文代码](https://github.com/Andrew-Qibin/CoordAttention)





## BiFPN特征融合

>[Cite]Tan, Mingxing, Ruoming Pang, and Quoc V. Le. “Efficientdet: Scalable and efficient object detection.” Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
>
>[论文地址](https://arxiv.org/abs/1911.09070)
>
>[论文代码](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)





## SwinTrans





## YOLOv5+Ghostconv+BiFPN+CA









## Results

| Models | mAP@.5 | mAP@.5:.95 | GFLOPS | params(M) | FLOPs | speed |
| ------ | ------ | ---------- | ------ | --------- | ----- | ----- |
|        |        |            |        |           |       |       |

## TO DO

+ [x] backbone: ShuffleNetV2
+ [x] backbone: Mobilenetv3
+ [x] backbone: Ghostnet
+ [ ] head:BiFPN
+ [ ] YOLOv5+Ghostconv+BiFPN+CA
+ [ ] backbone: SwinTrans
+ [ ] Prune: FSP

