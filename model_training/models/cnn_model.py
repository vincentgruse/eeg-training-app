"""
===============================================================================
EEGCNN: Convolutional Neural Network for EEG Motor Imagery Classification
===============================================================================

This CNN model is designed for EEG data shaped as 2D time-channel matrices.
The architecture is tailored to learn:
- Temporal patterns within each channel (via temporal convolution)
- Spatial correlations across channels (via spatial convolution)

Input Shape:
    [batch, 1, 64, 641]
    → where 64 = EEG channels, 641 = time samples

Model Architecture:
    - Conv2D temporal filter (1 × 15) across the time axis
    - Batch normalization + LeakyReLU activation
    - Conv2D spatial filter (64 × 1) across EEG channels
    - Global average pooling (1 × 20)
    - Dropout for regularization
    - Fully connected layer for binary classification (left vs. right)

Output:
    [batch, 2] class logits

Used in:
    train/train_cnn.py
-------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN(nn.Module):
    def __init__(self):
        super(EEGCNN, self).__init__()
        
        # First convolution:
        # Applies a temporal kernel of size 15 across time axis only (1 × 15)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolution:
        # Applies a spatial kernel of size 64 (across all EEG channels)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(64, 1))
        self.bn2 = nn.BatchNorm2d(32)
        
        # Adaptive pooling: reduces the time dimension to a fixed size of 20
        self.pool = nn.AdaptiveAvgPool2d((1, 20))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Final classification layer: flattens and maps to 2 output classes
        self.fc = nn.Linear(32 * 1 * 20, 2)  # Binary classification

    def forward(self, x):
        """
        Forward pass for EEGCNN.

        Args:
            x (Tensor): Input EEG tensor of shape [batch, 1, 64, 641]

        Returns:
            out (Tensor): Logits for [batch, 2] classes (left vs. right)
        """
        # Apply temporal convolution + BN + LeakyReLU
        x = F.leaky_relu(self.bn1(self.conv1(x)))       # shape -> [batch, 16, 64, 641]
        
        # Apply spatial convolution + BN + LeakyReLU
        x = F.leaky_relu(self.bn2(self.conv2(x)))       # shape -> [batch, 32, 1, 641]
        
        # Apply pooling to reduce time dimension to 20
        x = self.pool(x)                                # shape -> [batch, 32, 1, 20]
        x = self.dropout(x)
        
        # Flatten before fully connected layer
        x = x.view(x.size(0), -1)                       # Flatten
        
        # Final classification layer
        x = self.fc(x)                                  # shape [batch, 2]
        return x