import argparse
import torch
import logging

import numpy as np
from tqdm import tqdm 
from SimpleNN.model import EmbedFFNN, HashFFNN
from SimpleNN.data import CriteoDataset, BatchIterator
from sklearn.feature_extraction import FeatureHasher

def run_test_set(model, test_set, batch_size, enable_cuda, hasher):
    model.eval()
    with torch.no_grad():
        R = 0
        C = 0
        N = 0

        for sample, click, propensity in BatchIterator(test_set, batch_size, enable_cuda, hasher):
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

# def calc_loss(output_tensor, click_tensor, propensity_tensor, enable_cuda):
#     current_batch_size = output_tensor.shape[0]
#     click_tensor = click_tensor.view(-1, 1)
#     clicked = torch.zeros(current_batch_size, 1)
#     not_clicked = torch.ones(current_batch_size, 1)
#     if enable_cuda:
#         clicked = clicked.cuda()
#         not_clicked = not_clicked.cuda()
#     rectified_label = torch.where((click_tensor == 1), clicked, not_clicked)

#     clicked_tensor = torch.ones(current_batch_size, 1)
#     not_clicked_tensor = torch.ones(current_batch_size, 1) * 10
#     if enable_cuda:
#         clicked_tensor = clicked_tensor.cuda()
#         not_clicked_tensor = not_clicked_tensor.cuda()
#     o_tensor = torch.where((click == 1), not_clicked_tensor, clicked_tensor)
#     N_tensor = o_tensor.sum()

#     # Calculate R
#     R_tensor = (rectified_label * (output[:, 0, 0] / propensity_tensor).view(-1, 1)).sum() * 10 ** 4
#     loss_tensor = -(R_tensor / N_tensor)

#     return torch.sum(loss_tensor)

def calc_loss(output_tensor, click_tensor, propensity_tensor, lamb, enable_cuda):
    current_batch_size = output_tensor.shape[0]
    # click_tensor = click_tensor.view(-1, 1)
    # clicked = torch.zeros(current_batch_size, 1)
    # not_clicked = torch.ones(current_batch_size, 1)
    # if enable_cuda:
    #     clicked = clicked.cuda()
    #     not_clicked = not_clicked.cuda()
    # rectified_label = torch.where((click_tensor == 1), clicked, not_clicked)

    clicked_tensor = torch.ones(current_batch_size, 1)
    not_clicked_tensor = torch.ones(current_batch_size, 1) * 10
    if enable_cuda:
        clicked_tensor = clicked_tensor.cuda()
        not_clicked_tensor = not_clicked_tensor.cuda()
    o_tensor = torch.where((click == 1), not_clicked_tensor, clicked_tensor)
    N_tensor = o_tensor.sum()

    # Calculate R
    R_tensor = ((click_tensor - lamb) * (output[:, 0, 0] / propensity_tensor).view(-1, 1)).sum() * 10 ** 4
    return torch.sum(R_tensor) / torch.sum(N_tensor)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', default='data/vw_compressed_train')
    parser.add_argument('--test', default='data/vw_compressed_test')
    parser.add_argument('--lamb', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stop_idx', type=int, default=500000)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--feature_dict', type=str, default='data/feature_to_keys.json')
    parser.add_argument('--cuda', action='store_true')

    # If hashing is used the model needs to be changed
    parser.add_argument('--hashing', action='store_true')
    parser.add_argument('--model', default="EmbedFFN", choices=["EmbedFFN", "HashFNN"])
    parser.add_argument('--n_features', default=2**12, type=int)
    args = parser.parse_args()

    if not args.hashing:
        hasher = None
        logging.info("Loading training dataset.")
        train_set = CriteoDataset(args.train, args.feature_dict, args.stop_idx, args.start_idx)
        logging.info("Finished loading training dataset, loading testing dataset now.")
        test_set = CriteoDataset(args.test, args.feature_dict, args.stop_idx, args.start_idx)
        logging.info("Finished loading testing datset, initialising model now.")
        model = EmbedFFNN(args.embedding_dim, args.hidden_dim, train_set.feature_dict, args.cuda)
    else:
        hasher = FeatureHasher(n_features=args.n_features, input_type="dict")
        logging.info("Loading training dataset.")
        train_set = CriteoDataset(args.train, args.feature_dict, args.stop_idx, args.start_idx, hashing=True)
        logging.info("Finished loading training dataset, loading testing dataset now.")
        test_set = CriteoDataset(args.test, args.feature_dict, 30000000+args.stop_idx, 30000000, hashing=True)
        logging.info("Finished loading testing datset, initialising model now.")
        model = HashFFNN(args.n_features)        

    if args.cuda and torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    epoch_losses = []
    logging.info("Initialized dataset")
    run_test_set(model, test_set, args.batch_size, args.cuda, hasher)

    for i in range(args.epochs):
        logging.info("Starting epoch {}".format(i))

        losses = []
        for sample, click, propensity in tqdm(BatchIterator(train_set, args.batch_size, args.cuda, hasher)):
            optimizer.zero_grad()
            output = model(sample)
            loss = calc_loss(output, click, propensity, args.lamb, args.cuda)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        logging.info("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))

        run_test_set(model, test_set, args.batch_size, args.cuda, hasher)
