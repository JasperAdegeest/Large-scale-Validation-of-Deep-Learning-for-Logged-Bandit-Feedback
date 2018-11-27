from torch import nn
import torch

class EmbedFFNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, feature_dict, enable_cuda):
        super(EmbedFFNN, self).__init__()

        # Embedding layers
        self.embedding_size = embedding_size
        self.embedding_layers = []
        for i in range(3, 36):
            self.embedding_layers.append(
                nn.Embedding(len(feature_dict[str(i)]), embedding_size)
            )

        if enable_cuda:
            for i in range(33):
                self.embedding_layers[i] = self.embedding_layers[i].cuda()

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

class HashFFNN(nn.Module):
    def __init__(self, n_features):
        super(HashFFNN, self).__init__()

        # Embedding layers
        self.linear = nn.Linear(n_features, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_vector):
        score = self.linear(feature_vector)
        probability = self.softmax(score)
        return probability