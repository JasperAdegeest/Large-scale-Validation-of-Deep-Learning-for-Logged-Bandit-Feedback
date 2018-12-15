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
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, dropout, **kwargs):
        super(SmallEmbedFFNN, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        self.linear1 = nn.Linear(35 * embedding_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

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

        out = self.dropout(torch.cat(input, dim=2))
        out = self.dropout(self.linear1(out))
        out = self.relu(out)
        out = self.dropout(self.linear2(out))
        out = self.relu(out)
        out = self.linear3(out)
        return self.softmax(out)


class LargeEmbedFFNN(EmbedFFNN):
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, dropout, **kwargs):
        super(LargeEmbedFFNN, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        self.linear1 = nn.Linear(35 * embedding_dim, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

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

        out = self.dropout(torch.cat(input, dim=2))
        out = self.dropout(self.linear1(out))
        out = self.relu(out)
        out = self.dropout(self.linear2(out))
        out = self.relu(out)
        out = self.dropout(self.linear3(out))
        out = self.relu(out)
        out = self.linear4(out)
        return self.softmax(out)


class TinyEmbedFFNN(EmbedFFNN):
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, dropout, **kwargs):
        super(TinyEmbedFFNN, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        self.linear1 = nn.Linear(35 * embedding_dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout) 

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

        out = self.dropout(torch.cat(input, dim=2))
        out = self.dropout(self.linear1(out))
        out = self.relu(out)
        out = self.linear2(out)
        return self.softmax(out)


class HashFFNN2(nn.Module):
    def __init__(self, n_features):
        super(HashFFNN2, self).__init__()

        # Embedding layers
        #self.linear = nn.Linear(n_features, 1,  bias=False)
        self.linear1 = nn.Linear(n_features, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_vector):
        out = torch.spmm(feature_vector, self.linear1.weight.t()) + self.linear1.bias
        out = out.unsqueeze(0)
        out = self.relu(out)
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        score = self.linear4(out)
        probability = self.softmax(score)
        return probability



class HashFFNN(nn.Module):
    def __init__(self, n_features):
        super(HashFFNN, self).__init__()

        weights = torch.FloatTensor(n_features, 1)
        torch.nn.init.xavier_uniform_(weights)
        self.linear = nn.Parameter(weights)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_vector):
        score = torch.spmm(feature_vector, self.linear)
        score = score.unsqueeze(0)
        probability = self.softmax(score)
        return probability



class CrossLayer(nn.Module):
    def __init__(self, dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Linear(dim, 1, bias=False)
        self.bias = nn.Linear(dim, 1).bias

    def forward(self, x, x_0):
        
        correlated = self.weight(x)
        correlated.transpose(1, 2)
        x_0 = x_0.unsqueeze(3)
        correlated = torch.einsum('ijkl,ijl->ijk', (x_0, correlated))
        return correlated + self.bias + x

class CrossNetwork(EmbedFFNN):
    def __init__(self, feature_dict, device, embedding_dim, hidden_dim, enable_cuda, **kwargs):
        super(CrossNetwork, self).__init__(feature_dict, device, embedding_dim, enable_cuda)
        
        # Cross Network
        self.cross_layer1 = CrossLayer(embedding_dim*35)
        self.cross_layer2 = CrossLayer(embedding_dim*35)
        self.cross_layer3 = CrossLayer(embedding_dim*35)

        # Regular Deep Neural Network
        self.dnn_layer1 = nn.Linear(35 * embedding_dim, 1024)
        self.dnn_layer2 = nn.Linear(1024, 768)
        self.dnn_layer3 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.final_layer = nn.Linear(512 + embedding_dim*35, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_dim, pool_size, _ = x.shape
        embedded = []
        for i in range(35):
            if i < 2:
                tensor = x[:, :, i].unsqueeze(2)
                tensor = tensor.repeat(1, 1, self.embedding_dim)
                embedded.append(tensor)
            else:
                tensor = self.embedding_layers[i-2](x[:, :, i].long())
                embedded.append(tensor)
        embedded = torch.cat(embedded, dim=2)
        x_0 = embedded
        x = self.cross_layer1(x_0, x_0)
        x = self.cross_layer2(x, x_0)
        x_cross = self.cross_layer3(x, x_0)
        
        x = self.relu(self.dnn_layer1(x_0))
        x = self.relu(self.dnn_layer2(x))
        x_dnn = self.relu(self.dnn_layer3(x))

        out = self.final_layer(torch.cat((x_cross, x_dnn), dim=2))
        return self.softmax(out)

