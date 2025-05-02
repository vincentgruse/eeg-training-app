import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNNLSTM(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden_size=64, lstm_layers=1):
        super(EEGCNNLSTM, self).__init__()

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),  # [B, 1, 8, 250] → [B, 16, 8, 250]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),                      # [B, 16, 8, 125]

            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),  # [B, 32, 8, 125]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))                       # [B, 32, 8, 62]
        )

        # LSTM input: flatten spatial features into time sequence
        self.lstm_input_size = 32 * 8  # channels × spatial height
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
        # x: [B, 1, 8, 250]
        x = self.cnn(x)  # [B, 32, 8, 62]
        x = x.permute(0, 3, 1, 2)  # [B, 62, 32, 8] → sequence of 62 steps
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, 62, 256]

        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))  # [B, 62, hidden*2]
        out = out[:, -1, :]  # last time step
        out = self.dropout(out)
        return self.fc(out)
