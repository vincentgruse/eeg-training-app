"""
lstm.py

This file defines an LSTM (Long Short-Term Memory) network for classifying EEG
signals into motor intention classes such as FORWARD, STOP, etc.

Dependencies:
    torch
"""

import torch
import torch.nn as nn

class EEGLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=5, bidirectional=True):
        """
        Initialize the LSTM model for EEG classification.

        Args:
            input_size (int): Number of EEG channels (features per timestep)
            hidden_size (int): Number of LSTM hidden units
            num_layers (int): Number of stacked LSTM layers
            num_classes (int): Number of output classes
            bidirectional (bool): Whether to use a bidirectional LSTM
        """
        super(EEGLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM for sequential modeling. 250 timesteps.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output layer.
        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_out_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the LSTM.

        Args:
            x (Tensor): Input of shape [B, 250, 8]

        Returns:
            Tensor: Output logits of shape [B, num_classes]
        """
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))  # [batch, seq_len, hidden]
        out = out[:, -1, :]  # take last time step's output
        out = self.dropout(out)
        return self.fc(out)