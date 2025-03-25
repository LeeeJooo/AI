import torch
import torch.nn as nn
import torch.optim as optim

# data
import glob # 파일 시스템에서 특정 패턴에 맞는 파일 경로 목록을 검색할 때 사용. 와일드카드 사용 가능.
import os
import string
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")