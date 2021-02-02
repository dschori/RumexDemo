# Rumex Demo

## Data

**Structure:**

Data has to be placed like:
```
RumexDemo
│
└───images
│   │   img_1000.png
│   │   img_1001.png
│   │   ...
│   
└───masks
│   │   img_1000.png
│   │   img_1001.png
│   │   ...
```

## Training

Training Notebook: [Training.ipynb](https://github.com/dschori/RumexDemo/blob/master/Training.ipynb)

### Config

Configs can be made in [config.py](utils/config.py)

## Inference

Simple Inference: [Inference.ipynb](https://github.com/dschori/RumexDemo/blob/master/Inference.ipynb)

## Docker

Image hosted on: https://hub.docker.com/r/dschori/rumex_demo

**Build:**  
`docker build -t dschori/rumex_demo:latest -f Dockerfile .`

**Run**  
killer machine command:  

`docker run --rm -ti -e NVIDIA_VISIBLE_DEVICES=15 -p 8884:8888 -p 8888:6006 -v /mnt/data/dschori/rumex-workspace/RumexDemo/:/workspace/RumexDemo/data dschori/rumex_demo:latest bash`
