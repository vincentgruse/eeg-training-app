"""
cnn_lstm.py

This file defines a hybrid deep learning architecture that combines a 2D CNN
with an LSTM. It is used to classify EEG signals into motor intention categories
such as FORWARD, BACKWARD, or STOP.

Dependencies:
    torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNNLSTM(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden_size=64, lstm_layers=1):
        """
        Initialize CNN-LSTM hybrid model.

        Args:
            num_classes (int): Number of output movement classes (e.g., 2 or 5)
            lstm_hidden_size (int): Number of hidden units in LSTM layer
            lstm_layers (int): Number of LSTM layers
        """
        super(EEGCNNLSTM, self).__init__()

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        # LSTM input: flattens spatial features into time sequence.
        self.lstm_input_size = 32 * 8
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)  # bidirectional

    def forward(self, x):
        """
        Forward pass through the CNN-LSTM model.

        Args:
            x (Tensor): EEG window input of shape [B, 1, 8, 250]

        Returns:
            Tensor: Predicted class logits of shape [B, num_classes]
        """
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)

        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)