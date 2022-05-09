yolov5s

![image-20220503075136814](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202205030751864.png)





```bash
LD_PRELOAD=./libmyplugins.so deepstream-app -c deepstream_app_config_yoloV5.txt
```



```
import requests
import wandb
```

/root/anaconda3/bin/python

pip install -r requirements.txt

timm



找一篇相关的外文文献翻译





# baseline

```bash
nohup python train.py --weights yolov5s.pt --data data/helmet.yaml --epochs 300 --device 0 --name baseline --adam  &
```





# 稀疏化

## 参数为0

```bash
# sparity-0
nohup python train_sparity.py --st --sr 0 --weights yolov5s.pt --data data/helmet.yaml --epochs 100 --imgsz 640 --device 0 --name sparse-baseline --adam &
```





## 正常稀疏

```bash
# sparity-0.0002
nohup python train_sparity.py --st --sr 0.0002 --weights yolov5s.pt --data data/helmet.yaml --epochs 300 --imgsz 640 --device 1 --name sparse --adam &
```

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202205091427649.png" alt="image-20220509142750622" style="zoom: 33%;" />



## 参数过大

```bash
# sparity-0.001
nohup python train_sparity.py --st --sr 0.001 --weights yolov5s.pt --data data/helmet.yaml --epochs 100 --imgsz 640 --device 0 --name sparse-large --adam &
```

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202205091428948.png" alt="image-20220509142843915" style="zoom:33%;" />



## 参数小

```bash
# sparity-0.0001
nohup python train_sparity.py --st --sr 0.0001 --weights yolov5s.pt --data data/helmet.yaml --epochs 100 --imgsz 640 --device 1 --name sparse-smal --adam &
```

<img src="https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202205091426401.png" alt="image-20220509142657352" style="zoom: 33%;" />



## 参数特别小

```bash
nohup python train_sparity.py --st --sr 0.00005 --weights yolov5s.pt --data data/helmet.yaml --epochs 100 --imgsz 640 --device 1 --name sparse-s --adam &
```

