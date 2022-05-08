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





# sparity-0.0002

```bash
--st --sr 0.0002 --weights yolov5s.pt --data data/helmet.yaml --epochs 300 --imgsz 640 --device 1 --adam
```





