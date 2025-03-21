import os
import torch
import numpy as np
import torch.nn as nn
from constants import *
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = './models'