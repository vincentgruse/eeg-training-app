"""
===============================================================================
EEGCNNLSTM: Hybrid CNN + LSTM Model with Attention for EEG Motor Imagery
===============================================================================

This architecture combines the spatial-temporal strengths of CNNs and LSTMs 
for EEG signal classification (left vs. right hand motor imagery). It extracts 
spatial and frequency-related patterns via convolutional layers, followed by 
temporal sequence modeling using a bidirectional LSTM, enhanced with an 
attention mechanism to focus on the most relevant time segments.

Key Components:
- Conv2D layers for spatial (channel-wise) and temporal feature extraction
- Adaptive pooling to reduce dimensionality and normalize time axis
- LSTM for modeling sequential dependencies across extracted temporal features
- Attention mechanism to weigh important LSTM outputs
- Dropout + fully connected layer for final classification

Input Shape:
    [batch, 1, 64, 641] → EEG data as 2D input per channel

Output:
    [batch, 2] → logits for binary classification (left vs. right)

Used in:
    train/train_cnn_lstm.py
-------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism over LSTM outputs:
    Learns to assign weights to each time step and compute a context vector
    based on the most informative LSTM outputs.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): LSTM output of shape [batch, time, hidden*2]
        
        Returns:
            context (Tensor): Weighted sum of LSTM outputs [batch, hidden*2]
        """
        attn_weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context

class EEGCNNLSTM(nn.Module):
    """
    Hybrid CNN + LSTM + Attention model for EEG classification.
    """
    def __init__(self, lstm_hidden=128, proj_size=16):
        super(EEGCNNLSTM, self).__init__()

        # --- CNN Block ---
        # Temporal convolution across time within each channel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(16)
        
        # Spatial convolution across channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(64, 1))
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling to reduce temporal length and get consistent time steps
        self.pool = nn.AdaptiveAvgPool2d((1, 100))  # more time steps

        # --- LSTM Block ---
        # Project each time step's features down before LSTM
        self.proj = nn.Linear(32, proj_size)  # compress features per step
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention over LSTM outputs
        self.attn = Attention(lstm_hidden)
        
        # Final classification head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden * 2, 2)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input EEG tensor [batch, 1, 64, 641]
            
        Returns:
            out (Tensor): Logits [batch, 2]
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))    # [B, 16, 64, 641]
        x = F.leaky_relu(self.bn2(self.conv2(x)))    # [B, 32, 1, 641]
        x = self.pool(x)                             # [B, 32, 1, 100]
        
        # Reshape: squeeze channel, permute time to front
        x = x.squeeze(2).permute(0, 2, 1)            # [B, 100, 32]

        # Project each timestep’s feature vector to a smaller size
        x = self.proj(x)                             # [B, 100, proj_size]
        
        # Pass through LSTM
        x, _ = self.lstm(x)                          # [B, 100, hidden*2]
        
        # Apply attention over time
        x = self.attn(x)                             # [B, hidden*2]
        x = self.dropout(x)
        x = self.fc(x)                               # [B, 2]
        return x
