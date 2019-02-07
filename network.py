# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self, img_height):
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,300))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4,300))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,300))
        
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        
        self.fc = nn.Linear(in_features=img_height - 3, out_features=6)
        
    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(x))
        conv3_out = F.relu(self.conv3(x))
        
        concat_out = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)
        
        output = F.softmax(self.fc(concat_out))
        
        return output