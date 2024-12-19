'''
@brief Mobile-Net architecture, suitable for mobile processes.
@author Ayush Tripathi (atripathi7783@gmail.com)
'''

import torch.nn as nn
from src.back_end.model_training.arch.interface import *



'''
DepthWiseSeperable is a class that decomposes typical convolution into depthwise and pointwise convolutions, requiring less resources as matrices are smaller in dimension. Adapted from Karunesh Upadhyay's description. 
'''

class DepthWiseSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        DepthWiseSeperable block for MobileNet.
        """
        super(DepthWiseSeperable, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, 
            kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module, CVModel):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        # Initial convolution layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Depthwise separable convolutions
            DepthWiseSeperable(32, 64, 1),
            DepthWiseSeperable(64, 128, 2),
            DepthWiseSeperable(128, 128, 1),
            DepthWiseSeperable(128, 256, 2),
            DepthWiseSeperable(256, 256, 1),
            DepthWiseSeperable(256, 512, 2),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 1024, 2),
            DepthWiseSeperable(1024, 1024, 1),
        )

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )
    
    def forward():
        pass

    def train_model():
        pass
    
    def test_model():
        pass
    
    def predict():
        pass

    def save():
        pass