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

- **Purple:** Detected Rumex
- **Pink:** Detected Garbage

![Test Image 1](https://github.com/dschori/RumexDemo/blob/master/data/test1.PNG)
![Test Image 2](https://github.com/dschori/RumexDemo/blob/master/data/test2.PNG)

## Inference

**Trained Model:** [PyTorch Model](https://github.com/dschori/RumexDemo/tree/master/models/best_model.pth)

For Inference, the **image** has to be in shape **[Channels, Height, Width] -> [3, 768, 768]**  

Model returns for each image a **mask** in shape **[2, 768, 768]** -> first channel is rumex, second is garbage

For Example:  

```python
def image_to_torch_tensor(image):
    return torch.from_numpy(image).to('gpu').unsqueeze(0)

def torch_tensor_to_image(tensor):
    return tensor.squeeze().cpu().numpy().round()

tensor = image_to_torch_tensor(image_as_numpy)
pr_mask_as_tensor = model.predict(tensor)
pr_mask_as_numpy = torch_tensor_to_image(pr_mask_as_tensor)
```

## Docker

Dockerfile: [Dockerfile](https://github.com/dschori/RumexDemo/blob/master/docker/Dockerfile)

**Build:**  
`docker build -t dschori/rumex_demo:latest -f Dockerfile .`

**Run**  
killer machine command:  

`docker run --rm -ti -e NVIDIA_VISIBLE_DEVICES=15 -p 8884:8888 -p 8888:6006 -v /mnt/data/dschori/rumex-workspace/RumexDemo/:/workspace/RumexDemo/data dschori/rumex_demo:latest bash`
