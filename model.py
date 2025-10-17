import os 
import numpy as np 
import sys 
import math 
import torch 
import torch.nn.functional as F

from torch import nn 

class CustomModel(nn.Module):
    def __init__(self, num_classes:int):
        super(CustomModel, self).__init__()
        # 卷积层1: 输入3通道，输出16通道，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化，尺寸减半
        
        # 卷积层2: 输入16通道，输出32通道
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 卷积层3: 输入32通道，输出64通道
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接层
        # 经过3次池化，64x64 -> 8x8 (64 / 2^3 = 8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 全连接层1
        self.dropout = nn.Dropout(0.5)  # Dropout防止过拟合
        self.fc2 = nn.Linear(128, num_classes)  # 输出5类
        
    def forward(self, x):
        # x: [batch, 3, 64, 64]
        
        # 卷积 -> BN -> ReLU -> 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [batch, 16, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [batch, 32, 16, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [batch, 64, 8, 8]
        
        # 展平
        x = x.view(-1, 64 * 8 * 8)  # [batch, 64*8*8]
        
        # 全连接层
        x = F.relu(self.fc1(x))  # [batch, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, 5]

        return x
    

    
class FCNet(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()

        # 输入尺寸：3 * 64 * 64 = 12288
        self.fc1 = nn.Linear(3 * 64 * 64, 512)  # 第一全连接层
        self.bn1 = nn.BatchNorm1d(512)  # 批归一化
        self.dropout1 = nn.Dropout(0.5)  # Dropout防止过拟合
        
        self.fc2 = nn.Linear(512, 128)  # 第二全连接层
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, num_classes)  # 输出层
        
    def forward(self, x):
        # x: [batch, 3, 64, 64]
        
        # 展平输入
        x = x.view(-1, 3 * 64 * 64)  # [batch, 12288]
        
        # 全连接层 -> BN -> ReLU -> Dropout
        x = F.relu(self.bn1(self.fc1(x)))  # [batch, 512]
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))  # [batch, 128]
        x = self.dropout2(x)
        
        x = self.fc3(x)  # [batch, 5]
        
        return x