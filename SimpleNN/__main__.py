import argparse
import torch
import logging

import numpy as np
from SimpleNN.model import SimpleNN
from SimpleNN.data import CriteoDataset, BatchIterator


def run_test_set(model, test_set, batch_size, enable_cuda):
    model.eval()
    with torch.no_grad():
        R = 0
        C = 0
        N = 0

        for sample, click, propensity in BatchIterator(test_set, batch_size, enable_cuda):
            output = model(sample)
            current_batch_size = output.shape[0]
            click = click.view(-1, 1)
            not_clicked_tensor = torch.zeros(current_batch_size, 1)
            clicked_tensor = torch.ones(current_batch_size, 1)
            if enable_cuda:
                not_clicked_tensor = not_clicked_tensor.cuda()
                clicked_tensor = clicked_tensor.cuda()

            rectified_label = torch.where((click == 1),
                              not_clicked_tensor,
                              clicked_tensor)

            clicked_tensor = torch.ones(current_batch_size, 1)
            not_clicked_tensor = torch.ones(current_batch_size, 1) * 10
            if enable_cuda:
                clicked_tensor = clicked_tensor.cuda()
                not_clicked_tensor = not_clicked_tensor.cuda()
            o = torch.where((click == 1), not_clicked_tensor, clicked_tensor)
            N += o.sum()

            # Calculate R
            R += (rectified_label * (output[:, 0, 0] / propensity).view(-1, 1)).sum()

            # Calculate C
            C += ((output[:, 0, 0] / propensity).view(-1, 1)).sum()

        R = (R / N) * 10**4
        C = C / N
        R_div_C = R / C

        print("\nTest results:")
        print("R x 10^4: {}\t C: {}\t (R x 10^4) / C: {}\n".format(R, C, R_div_C))


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', default='data/vw_compressed_train')
    parser.add_argument('--test', default='data/vw_compressed_test')
    parser.add_argument('--lamb', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stop_idx', type=int, default=50000)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--feature_dict', type=str, default='data/feature_to_keys.json')
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()
    logging.info("Loading training dataset.")
    train_set = CriteoDataset(args.train, args.feature_dict, args.stop_idx)
    logging.info("Finished loading training dataset, loading testing dataset now.")
    test_set = CriteoDataset(args.test, args.feature_dict, args.stop_idx)
    logging.info("Finished loading testing datset, initialising model now.")
    model = SimpleNN(args.embedding_dim, args.hidden_dim, train_set.feature_dict, args.cuda)

    if args.cuda and torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    epoch_losses = []
    print("Initialized dataset")
    run_test_set(model, test_set, args.batch_size, args.cuda)

    for i in range(args.epochs):
        print("Starting epoch {}".format(i))

        losses = []
        for sample, click, propensity in BatchIterator(train_set, args.batch_size, args.cuda):
            optimizer.zero_grad()
            output = model(sample)
            loss = -1 * (click - args.lamb) * (output[:, 0, 0] / propensity)
            loss = torch.mean(loss)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        print("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))

        run_test_set(model, test_set, args.batch_size, args.cuda)
