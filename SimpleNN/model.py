from torch import nn
import torch

class SimpleNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, feature_dict):
        super(SimpleNN, self).__init__()

        # Embedding layers
        self.embedding_size = embedding_size
        self.embedding_layers = []
        for i in range(3, 36):
            self.embedding_layers.append(
                nn.Embedding(len(feature_dict[str(i)]), embedding_size)
            )

        self.linear1 = nn.Linear(35 * embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_dim, pool_size, _ = x.shape
        input = []
        for i in range(35):
            if i < 2:
                tensor = x[:, :, i].unsqueeze(2)
                tensor = tensor.repeat(1, 1, self.embedding_size)
                input.append(tensor)
            else:
                tensor = self.embedding_layers[i-2](x[:, :, i].long())
                input.append(tensor)

        out = torch.cat(input, dim=2)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return self.softmax(out)