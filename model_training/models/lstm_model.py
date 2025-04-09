"""
===============================================================================
EEGLSTM: LSTM Model with Attention for EEG Motor Imagery Classification
===============================================================================

This model defines a bidirectional LSTM architecture with an attention mechanism
to classify motor imagery EEG data (left vs. right hand imagery).

Key Components:
- A projection layer that reduces the input feature dimensionality
- A 2-layer bidirectional LSTM to capture temporal dependencies across EEG timepoints
- An attention mechanism to weigh the importance of each timestep in the LSTM output
- A final classification layer with dropout for regularization

Input Shape:
    [batch, 641, 64] → where 641 is the number of time samples and 64 is the number of EEG channels

Output:
    [batch, 2] → logits for left-hand and right-hand motor imagery

Usage:
    Used in conjunction with train/train_lstm.py
-------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism that computes a context vector as a weighted sum
    over the LSTM output across the time dimension.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Learnable projection from [batch, seq_len, hidden*2] → [batch, seq_len, 1]
        self.attn = nn.Linear(hidden_size * 2, 1)  # for bidirectional LSTM

    def forward(self, lstm_output):
        """
        Args:
            lstm_output (Tensor): [batch, seq_len, hidden_size * 2]
            
        Returns:
            context (Tensor): [batch, hidden_size * 2], aggregated by attention weights
        """
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_output, dim=1)       # [batch, hidden*2]
        return context

class EEGLSTM(nn.Module):
    """
    LSTM-based classifier with projection, attention, and dropout for EEG data.
    """
    def __init__(self, input_size=64, proj_size=32, hidden_size=128, num_layers=2, num_classes=2):
        super(EEGLSTM, self).__init__()
        
        # Linear projection to reduce dimensionality from 64 to 32
        self.proj = nn.Linear(input_size, proj_size)

        # Bidirectional LSTM: processes the EEG sequence forward and backward
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Attention to focus on informative time steps in the sequence
        self.attention = Attention(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Final classification layer (fully connected)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): EEG input of shape [batch, 641, 64]
            
        Returns:
            out (Tensor): Output logits of shape [batch, 2]
        """
        x = self.proj(x)                     # [batch, 641, 32]
        out, _ = self.lstm(x)                # [batch, 641, hidden*2]
        out = self.attention(out)            # [batch, hidden*2]
        out = self.dropout(out)
        out = self.fc(out)                   # [batch, 2]
        return out
