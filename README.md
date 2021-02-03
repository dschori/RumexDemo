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

### Test Image Results:

- **Purple: Detected Rumex**
- **Pink: Detected Garbage**

![Test Image 1](https://github.com/dschori/RumexDemo/blob/master/data/test1.PNG)
![Test Image 2](https://github.com/dschori/RumexDemo/blob/master/data/test2.PNG)

## Docker

Dockerfile: [Dockerfile](https://github.com/dschori/RumexDemo/blob/master/docker/Dockerfile)

**Build:**  
`docker build -t dschori/rumex_demo:latest -f Dockerfile .`

**Run**  
killer machine command:  

`docker run --rm -ti -e NVIDIA_VISIBLE_DEVICES=15 -p 8884:8888 -p 8888:6006 -v /mnt/data/dschori/rumex-workspace/RumexDemo/:/workspace/RumexDemo/data dschori/rumex_demo:latest bash`
