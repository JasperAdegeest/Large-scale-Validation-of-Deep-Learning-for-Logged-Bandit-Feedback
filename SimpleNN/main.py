import argparse
import torch

from model import SimpleNN
from data import CriteoDataset, BatchIterator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train = CriteoDataset(args.train, 5000)
    model = SimpleNN(35, 100)
    optimizer = torch.optim.Adam(model.parameters())

    epoch_losses = []
    for i in range(args.epochs):
        print("Starting epoch {}".format(i))
        losses = []
        for sample, click, propensity in BatchIterator(train):
            optimizer.zero_grad()
            output = model(sample)
            loss = (click - args.lamb) * (output[0] / propensity)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        print("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))
