from torch import nn
import torch

class EmbedFFNN(nn.Module):
    def __init__(self, feature_dict, device, embedding_dim, enable_cuda):
        super(EmbedFFNN, self).__init__()

        # Embedding layers
        self.embedding_dim = embedding_dim
        self.embedding_layers = []
        for i in range(3, 36):
            self.embedding_layers.append(
                nn.Embedding(len(feature_dict[str(i)]), embedding_dim)
            )

        if enable_cuda:
            for i in range(33):
                self.embedding_layers[i] = self.embedding_layers[i].to(device)

    def forward(self, x):
        raise NotImplementedError()


class SmallEmbedFFNN(EmbedFFNN):
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, **kwargs):
        super(SmallEmbedFFNN, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        self.linear1 = nn.Linear(35 * embedding_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_dim, pool_size, _ = x.shape
        input = []
        for i in range(35):
            if i < 2:
                tensor = x[:, :, i].unsqueeze(2)
                tensor = tensor.repeat(1, 1, self.embedding_dim)
                input.append(tensor)
            else:
                tensor = self.embedding_layers[i-2](x[:, :, i].long())
                input.append(tensor)

        out = torch.cat(input, dim=2)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return self.softmax(out)


class LargeEmbedFFNN(EmbedFFNN):
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, **kwargs):
        super(LargeEmbedFFNN, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        self.linear1 = nn.Linear(35 * embedding_dim, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_dim, pool_size, _ = x.shape
        input = []
        for i in range(35):
            if i < 2:
                tensor = x[:, :, i].unsqueeze(2)
                tensor = tensor.repeat(1, 1, self.embedding_dim)
                input.append(tensor)
            else:
                tensor = self.embedding_layers[i-2](x[:, :, i].long())
                input.append(tensor)

        out = torch.cat(input, dim=2)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return self.softmax(out)


class TinyEmbedFFNN(EmbedFFNN):
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, **kwargs):
        super(TinyEmbedFFNN, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        self.linear1 = nn.Linear(35 * embedding_dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_dim, pool_size, _ = x.shape
        input = []
        for i in range(35):
            if i < 2:
                tensor = x[:, :, i].unsqueeze(2)
                tensor = tensor.repeat(1, 1, self.embedding_dim)
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
        #self.linear = nn.Linear(n_features, 1,  bias=False)
        weights = torch.FloatTensor(n_features, 1)
        torch.nn.init.xavier_uniform_(weights)
        self.linear = nn.Parameter(weights)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_vector):
        score = torch.spmm(feature_vector, self.linear)
        score = score.unsqueeze(0)
        probability = self.softmax(score)
        return probability