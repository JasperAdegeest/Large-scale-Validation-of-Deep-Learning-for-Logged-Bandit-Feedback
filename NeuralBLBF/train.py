import torch
import logging

import numpy as np
from tqdm import tqdm 

from NeuralBLBF.evaluate import run_test_set
from NeuralBLBF.data import BatchIterator


def calc_loss(output_tensor, click_tensor, propensity_tensor, lamb, enable_cuda):
    N_hat = torch.sum(click_tensor.eq(1) * 10 + click_tensor.eq(0)).float()
    R_hat = (click_tensor - lamb) * (output_tensor[:, 0, 0] / propensity_tensor)
    return torch.sum(R_hat) / torch.sum(N_hat)

def train(model, optimizer, train_set, test_set, batch_size, enable_cuda, epochs, lamb, hasher=None):
    epoch_losses = []
    logging.info("Initialized dataset")
    run_test_set(model, test_set, batch_size, enable_cuda, hasher)

    for i in range(epochs):
        logging.info("Starting epoch {}".format(i))

        losses = []
        for sample, click, propensity in tqdm(BatchIterator(train_set, batch_size, enable_cuda, hasher)):
            optimizer.zero_grad()
            output = model(sample)
            loss = calc_loss(output, click, propensity, lamb, enable_cuda)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        logging.info("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))

        run_test_set(model, test_set, batch_size, enable_cuda, hasher)

############ BIN ################

# def old_calc_loss(output_tensor, click_tensor, propensity_tensor, enable_cuda):
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
#     o_tensor = torch.where((click_tensor == 1), not_clicked_tensor, clicked_tensor)
#     N_tensor = o_tensor.sum()

#     # Calculate R
#     R_tensor = (rectified_label * (output_tensor[:, 0, 0] / propensity_tensor).view(-1, 1)).sum() * 10 ** 4
#     loss_tensor = -(R_tensor / N_tensor)

#     return torch.sum(loss_tensor)