import argparse
import torch

from model import SimpleNN
from data import CriteoDataset, BatchIterator
import numpy.random
import numpy as np

def run_test_set(model, test_set):
    model.eval()
    with torch.no_grad():
        R = []
        C = []

        for sample, click, propensity in BatchIterator(test_set):
            output = model(sample)

            # Calculate R
            if click == 0.999:
                o = 1.0
            else:
                o = float(numpy.random.choice([0, 10], p=[0.9, 0.1]))
            R.append(click * (output[0] / propensity) * o)

            # Calculate C
            if click == 0.999:
                o = 1
            else:
                o = float(numpy.random.choice([0, 10], p=[0.9, 0.1]))
            C.append((output[0] / propensity) * o)

        R = np.average(R) * 10**4
        C = np.average(C)
        R_div_C = R / C

        print("\nTest results:")
        print("R x 10^4: {}\t C: {}\t (R x 10^4) / C: {}\n".format(R, C, R_div_C))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', default='../data/vw_compressed_train')
    parser.add_argument('--lamb', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    train = CriteoDataset(args.train, 50000)
    model = SimpleNN(20, 100)
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

        run_test_set(model, train)
