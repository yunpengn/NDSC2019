import torch.nn as nn
from src.data import text_field


# Defines CNN topology
class Cnn(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
        super().__init__()
        self.embedding = nn.Embedding(84503, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []

        # Fully connected layers
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)

        # Output layer
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        return self.predictor(feature)
