# 消融实验

数据集：[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
train：4500
val：500

# 基础参数设置

batch-size: 32

Image-size: 640

# 预训练测试

- exp11

  models:yolov5n.yaml

  Model size:3.9M

  Pretrain: yolov5n

  batch-size: 32

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407093516688.png" alt="image-20220407093516688" width="800" />

- exp15

  models：yolov5n.yaml

  Model size:3.9M

  Pretrain：none
  batch-size：128

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407093558329.png" alt="image-20220407093558329" width="800"/>



# Shufflenetv2

[Cite]Ma, Ningning, et al. “Shufflenet v2: Practical guidelines for efficient cnn architecture design.” Proceedings of the European conference on computer vision (ECCV). 2018.

[论文地址](https://arxiv.org/abs/1807.11164)

[论文代码](https://github.com/megvii-model/ShuffleNet-Series)

- exp17

  models: yolov5-shufflenetv2.yaml

  Model size: 805kB

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/image-20220407093620656.png" alt="image-20220407093620656" width="800" />



# Mobilenetv3

[Cite]Howard, Andrew, et al. “Searching for mobilenetv3.” Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

[论文地址](https://arxiv.org/abs/1905.02244)

[论文代码](https://github.com/xiaolai-sqlai/mobilenetv3)

- exp18
- models: yolov5-mobilenetv3-small.yaml





# Ghostnet

Han, Kai, et al. “Ghostnet: More features from cheap operations.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[论文地址](https://arxiv.org/abs/1911.11907)

[论文代码](https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch)



