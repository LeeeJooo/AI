from config import *
from constants import *

# pooling 사용은 지양
# global average pooling (task 에 따라 필요하다면)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)     # 28 - 5 + 1 = 24   >> (batch_size, 20, 24, 24)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)    # 24 - 5 + 1 = 20    >> (batch_size, 50, 20, 20)
        
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)                                  # 50 x 20 x 20 = 20000  >> (batch_size, 20000)
        self.fc1 = nn.Linear(in_features=20*20*50, out_features=128)                          #                   >> (batch_size, 128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)                              #                   >> (batch_size, 10)
        
        self.softmax = nn.Softmax(dim=1)

    # def __init__(self):
    #     super(CNN,self).__init__()
    #     self.relu = nn.ReLU()

    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)     # 28 - 5 + 1 = 24   >> (batch_size, 20, 24, 24)
    #     self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                              # (24-2)/2 + 1 = 12 >> (batch_size, 20, 12, 12) 
        
    #     self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)    # 12 - 5 + 1 = 8    >> (batch_size, 50, 8, 8)
    #     self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                              # (8-2)/2 + 1 = 5   >> (batch_size, 50, 4, 4)
        
    #     self.flatten = nn.Flatten(start_dim=1, end_dim=-1)                                  # 50 x 4 x 4 = 800  >> (batch_size, 800)
    #     self.fc1 = nn.Linear(in_features=4*4*50, out_features=128)                          #                   >> batch_size, 128)
    #     self.fc2 = nn.Linear(in_features=128, out_features=10)                              #                   >> (batch_size, 10)
        
    #     self.softmax = nn.Softmax(dim=1)
                
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        # X = self.max_pool1(X)

        X = self.conv2(X)
        X = self.relu(X)
        # X = self.max_pool2(X)

        X = self.flatten(X)

        X = self.fc1(X)
        X = self.relu(X)
        
        X = self.fc2(X)
        X = self.softmax(X)

        return X