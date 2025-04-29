import torch
import torch.nn as nn
import random
import math

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class Network(nn.Module):
    def __init__(self, h, w, outputs):
        super(Network, self).__init__()
        '''
        input           -> BATCH_SIZE, 3, 4, 4
        CONV1           -> BATCH_SIZE, 8, 2, 2
        CONV2           -> BATCH_SIZE, 16, 2, 2
        LINEARIZATION   -> BATCH_SIZE, 64
        FC              -> BATCH_SIZE, 4
        '''
        CONV1_OUT_CHANNELS = 8
        CONV1_KERNEL_SIZE = 3
        CONV2_OUT_CHANNELS = 16
        CONV2_KERNEL_SIZE = 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=CONV1_OUT_CHANNELS, kernel_size=CONV1_KERNEL_SIZE, stride=1, bias=True) # BATCH_SIZE, 16, 252, 252
        self.bn1 = nn.BatchNorm2d(num_features=CONV1_OUT_CHANNELS, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=CONV2_KERNEL_SIZE, stride=1) # BATCH_SIZE, 32, 248, 248
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNELS)

        self.relu = nn.ReLU()

        def conv2d_size_out(size, kernel_size, stride=1):
            return (size-kernel_size)//stride + 1
        
        convh = conv2d_size_out(conv2d_size_out(h, CONV1_KERNEL_SIZE), CONV2_KERNEL_SIZE)
        convw = conv2d_size_out(conv2d_size_out(w, CONV1_KERNEL_SIZE), CONV2_KERNEL_SIZE)

        linear_input_size = convh * convw * CONV2_OUT_CHANNELS
        self.fc = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def evaluate(self):
        pass