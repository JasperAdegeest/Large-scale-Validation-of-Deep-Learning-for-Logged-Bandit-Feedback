from torch import nn
import torch
import numpy as np
from SimpleNN.config import Config


class SimpleNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(SimpleNN, self).__init__()
        config = Config()

        # Embedding layers
        self.embedding_size = embedding_size
        self.embedding_layers = []
        for i in range(3, 36):
            self.embedding_layers.append(nn.Embedding(config.get_feature_size(i), embedding_size))

        self.linear1 = nn.Linear(35 * embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        input = []
        for i in range(x.shape[1]):
            if i < 2:
                tensor = x[:, i].repeat(self.embedding_size, 1).t()
                input.append(tensor)
            else:
                tensor = self.embedding_layers[i-2](x[:, i].long())
                input.append(tensor)

        out = torch.cat(input, dim=1)

        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)

        return self.softmax(out)