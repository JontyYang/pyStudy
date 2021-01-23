import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image

# 导入预训练的Vgg16网络
vgg16 = models.vgg16(pretrained=True)
# 读取一张图像，并对其进行可视化
im = Image.open("data/image/elephant.jpg")

