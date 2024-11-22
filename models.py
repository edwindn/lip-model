import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(8, output_dim),
        )

        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        input, prev_output = x
        x = self.encoder(input)
        x = torch.cat((x, prev_output), axis=-1)
        coords = self.fc(x)
        return coords

class EncoderBatched(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(0.1),

            nn.Linear(24, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.05),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.05),

            nn.Linear(8, output_dim),
        )

        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        input, prev_output = x
        x = self.encoder(input)
        x = torch.cat((x, prev_output), axis=-1)
        coords = self.fc(x)
        return coords