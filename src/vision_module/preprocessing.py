import numpy as np
import torch
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def preprocess_image(img: np.ndarray) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return _transform(img).unsqueeze(0)