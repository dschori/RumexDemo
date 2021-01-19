# Rumex Demo

## Data

**Structure:**

Data has to be place like:
```
RumexDemo
│
└───images
│   │   img_1000.png
│   │   img_1001.png
│   │   ...
│   
└───masks
│   │   msk_1000.png
│   │   msk_1001.png
│   │   ...
```

## Training

### Config

Configs can be made in [config.py](config.py)

## Inference

TODO...

## Docker Commands

**Build:**  
`docker build -t dschori/rumex_demo:latest -f Dockerfile .`

**Run**  
`docker run -it -p 8888:8888 -p 6006:6006 dschori/rumex_demo:latest bash`