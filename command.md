



```
import requests
import wandb
```

```bash
pip install -r requirements.txt

/root/anaconda3/bin/python

pip install -r requirements.txt

pip install timm
```



##### Usage

1. 正常训练模型

```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/prunModels/yolov5s-pruning.yaml --device 0
```

（注意训练其他模型，参考/prunModels/yolov5s-pruning.yaml进行修改，目前已支持v6架构）

1. 搜索最优子网

```
python pruneEagleEye.py --weights path_to_trained_yolov5_model --cfg models/prunModels/yolov5s-pruning.yaml --data data/VisDrone.yaml --path path_to_pruned_yolov5_yaml --max_iter maximum number of arch search --remain_ratio the whole FLOPs remain ratio --delta 0.02
```

1. 微调恢复精度

```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights path_to_Eaglepruned_yolov5_model --cfg path_to_pruned_yolov5_yaml --device 0
```



## deepstream



```bash
LD_PRELOAD=./libmyplugins.so deepstream-app -c deepstream_app_config_yoloV5.txt
```

