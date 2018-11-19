from torch import nn
import torch
import numpy as np
from SimpleNN.Config import Config


class SimpleNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(SimpleNN, self).__init__()
        config = Config()

        # Embedding layers
        self.embedding_size = embedding_size
        self.embedding_layers = []
        for i in range(3, 36):
            self.embedding_layers.append(nn.Embedding(config.get_feature_size(i), embedding_size))

        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        input = []
        for i in range(len(x)):
            if i < 3:
                print(x[i])
                tensor = torch.Tensor(x[i]) * self.embedding_size
                input.extend(tensor)
            else:
                results = []
                for item in x[i]:
                    results.append(self.embedding_layers[i](item))
                input.extend(np.average(results))

        out = self.linear1(input)
        out = self.relu(out)
        out = self.linear2(out)

        return self.softmax(out)