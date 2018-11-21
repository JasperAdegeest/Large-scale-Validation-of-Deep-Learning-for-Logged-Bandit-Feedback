import argparse
import torch

from SimpleNN.model import SimpleNN
from SimpleNN.data import CriteoDataset, BatchIterator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stop_idx', type=int, default=5000)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--feature_dict', type=str, default='data/features_to_keys.json')
    args = parser.parse_args()
    train = CriteoDataset(args.train, args.feature_dict, args.stop_idx)
    model = SimpleNN(args.embedding_dim, args.hidden_dim, train.feature_dict)
    optimizer = torch.optim.Adam(model.parameters())

    epoch_losses = []
    for i in range(args.epochs):
        print("Starting epoch {}".format(i))
        losses = []
        for sample, click, propensity in BatchIterator(train, args.batch_size):
            optimizer.zero_grad()
            output = model(sample)
            loss = (click - args.lamb) * (output[:, 0, 0] / propensity)
            loss = torch.sum(loss)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        print("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))
