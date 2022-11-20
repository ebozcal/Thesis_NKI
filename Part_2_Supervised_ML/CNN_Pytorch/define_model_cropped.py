
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *


class cnn3d3(nn.Module):

    def __init__(self, nf,  dr, af):
        super(cnn3d3, self).__init__()

        self.nf = nf
        self.dr = dr
        self.af = af
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dr)

        self.conv1 = self._conv_layer_set(1, nf*2)
        self.conv2 = self._conv_layer_set(nf*2, nf*2*2)
        self.conv3 =self._conv_layer_set(nf*2*2, nf*2*2*2)
        self.fc1 = nn.Linear(nf*2*2*2*10*22*22, 128)
        self.fc2 = nn.Linear(128, 3)

        self.conv1_bn = nn.BatchNorm3d(nf*2)
        self.conv2_bn = nn.BatchNorm3d(nf*2*2)
        self.conv3_bn = nn.BatchNorm3d(nf*2*2*2)

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=0,
                ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        #print('input shape:', x.shape)
        x = self.conv1(x)
        #print("shape after conv1: " + str(x.shape))
        x = self.conv1_bn(x)
        x = self.conv2(x)
        #print("shape after conv2: " + str(x.shape))
        x = self.conv2_bn(x)
        x = self.conv3(x)
        #print("shape after conv3: " + str(x.shape))
        x = self.conv3_bn(x)
        #print("shape after conv4: " + str(x.shape))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        #if self.af=="sigmoid":
         #   x = F.sigmoid(x)
        #if self.af=="softmax":
         #   x = F.softmax(x, dim=-1)

        return x



class cnn3d4(nn.Module):

    def __init__(self, nf, dr, af):
        super(cnn3d4, self).__init__()

        self.nf = nf
        self.dr = dr
        self.af = af
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dr)

        self.conv1 = self._conv_layer_set(1, nf*2)
        self.conv2 = self._conv_layer_set(nf*2, nf*2*2)
        self.conv3 =self._conv_layer_set(nf*2*2, nf*2*2*2)
        self.conv4 = self._conv_layer_set(nf*2*2*2, nf*2*2*2*2)
        self.fc1 = nn.Linear(nf*2*2*2*2*4*10*15, 128)
        self.fc2 = nn.Linear(128, 3)

        self.conv1_bn = nn.BatchNorm3d(nf*2)
        self.conv2_bn = nn.BatchNorm3d(nf*2*2)
        self.conv3_bn = nn.BatchNorm3d(nf*2*2*2)
        self.conv4_bn = nn.BatchNorm3d(nf*2*2*2*2)

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=0,
                ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        print('input shape:', x.shape)
        x = self.conv1(x)
        print("shape after conv1: " + str(x.shape))
        x = self.conv1_bn(x)
        x = self.conv2(x)
        print("shape after conv2: " + str(x.shape))
        x = self.conv2_bn(x)
        x = self.conv3(x)
        print("shape after conv3: " + str(x.shape))
        x = self.conv3_bn(x)
        x = self.conv4(x)
        print("shape after conv4: " + str(x.shape))
        x = self.conv4_bn(x)
    
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        print("shape after fc1: " + str(x.shape))
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        #if self.af=="sigmoid":
         #   x = F.sigmoid(x)
        #if self.af=="softmax":
         #   x = F.softmax(x, dim=-1)

        return x


        
































