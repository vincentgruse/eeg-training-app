"""
cnn.py

This file defines a Convolutional Neural Network (CNN) architecture designed to
classify raw EEG windows into motor intent classes (e.g., FORWARD, STOP).

Dependencies:
    torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN(nn.Module):
    def __init__(self, num_classes=5):
        """
        Initialize the EEG CNN model for motor intention classification.

        Args:
            num_classes (int): Number of output classes (e.g., 2 or 5).
        """
        super(EEGCNN, self).__init__()
        
        ## Convolutional Blocks
        
        # First convolution.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        # Second convolution.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        # Third convolution.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (Tensor): EEG input of shape [B, 1, 8, 250]

        Returns:
            Tensor: Output logits of shape [B, num_classes]
        """
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)