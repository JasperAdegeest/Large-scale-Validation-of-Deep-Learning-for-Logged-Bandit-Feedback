import argparse
import torch

import numpy as np
from SimpleNN.model import SimpleNN
from SimpleNN.data import CriteoDataset, BatchIterator


def run_test_set(model, test_set, batch_size):
    model.eval()
    with torch.no_grad():
        # THIS IS HOW I THINK IT MUST BE
        # R = 0
        # R_N = 0
        # C = 0
        # C_N = 0
        #
        # for sample, click, propensity in BatchIterator(test_set, batch_size):
        #     output = model(sample)
        #     current_batch_size = output.shape[0]
        #     click = click.view(-1, 1)
        #
        #     clicked_tensor = torch.ones(current_batch_size, 1)
        #     not_clicked_tensor = torch.ones(current_batch_size, 1) * 10
        #     o = torch.where((click == 0.999), clicked_tensor, not_clicked_tensor)
        #
        #     # Calculate R
        #     R += (click * (output[:, 0, 0] * propensity).view(-1, 1) * o).sum()
        #     R_N += o.sum()
        #
        #     # Calculate C
        #     C += ((output[:, 0, 0] * propensity).view(-1, 1) * o).sum()
        #     C_N += o.sum()
        #
        # R = (R / R_N) * 10**4
        # C = C / C_N
        # R_div_C = R / C

        # THIS GETS AN APPROPRIATE R
        R = []
        C = []

        for sample, click, propensity in BatchIterator(test_set, batch_size):
            output = model(sample)
            click = click.view(-1, 1)

            weight = (output[:, 0, 0] * propensity).view(-1, 1)

            # Calculate R
            R.extend(click * weight)

            # Calculate C
            C.extend(weight)

        R = np.average(R) * 10 ** 4
        C = np.average(C)
        R_div_C = R / C

        print("\nTest results:")
        print("R x 10^4: {}\t C: {}\t (R x 10^4) / C: {}\n".format(R, C, R_div_C))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', default='../data/vw_compressed_train')
    parser.add_argument('--test', default='../data/vw_compressed_test')
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stop_idx', type=int, default=5000)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--feature_dict', type=str, default='../data/features_to_keys.json')
    args = parser.parse_args()
    train_set = CriteoDataset(args.train, args.feature_dict, args.stop_idx)
    test_set = CriteoDataset(args.test, args.feature_dict, args.stop_idx)
    model = SimpleNN(args.embedding_dim, args.hidden_dim, train_set.feature_dict)
    optimizer = torch.optim.Adam(model.parameters())

    epoch_losses = []
    for i in range(args.epochs):
        print("Starting epoch {}".format(i))

        losses = []
        run_test_set(model, test_set, args.batch_size)


        for sample, click, propensity in BatchIterator(train_set, args.batch_size):
            optimizer.zero_grad()
            output = model(sample)
            loss = (click - args.lamb) * (output[:, 0, 0] / propensity)
            loss = torch.sum(loss)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        print("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))

        run_test_set(model, test_set, args.batch_size)
