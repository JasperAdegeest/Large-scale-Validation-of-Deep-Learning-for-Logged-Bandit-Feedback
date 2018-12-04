import torch
import logging

import numpy as np
from tqdm import tqdm

from NeuralBLBF.data import CriteoDataset, BatchIterator


from NeuralBLBF.evaluate import run_test_set
from NeuralBLBF.data import BatchIterator, get_start_stop_idx, CriteoDataset


def calc_loss(output_tensor, click_tensor, propensity_tensor, lamb, gamma, enable_cuda):
    N_hat = torch.sum(click_tensor.eq(1) * 10 + click_tensor.eq(0)).float()
    R_hat = (click_tensor - lamb) * (output_tensor[:, 0, 0] / propensity_tensor)
    return torch.sum(R_hat) / torch.sum(N_hat)


def train(model, optimizer, feature_dict, device, save_model_path, train, test,
          batch_size, enable_cuda, epochs, lamb, gamma, sparse, stop_idx, step_size,
          save, **kwargs):
    epoch_losses = []
    logging.info("Initialized dataset")
    run_test_set(model, test, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device)

    for i in range(epochs):
        logging.info("Starting epoch {}".format(i))

        losses = []
        for j in range(0, stop_idx, step_size):

            logging.info("Loading training {} to {} out of {}.".format(j, j+step_size, stop_idx))
            train_set = CriteoDataset(train, feature_dict, j+step_size, j, sparse, save)
            if not sparse:
                for k, (sample, click, propensity) in enumerate(BatchIterator(train_set, batch_size, enable_cuda, sparse, device)):
                    optimizer.zero_grad()
                    output = model(sample)
                    loss = calc_loss(output, click, propensity, lamb, gamma, enable_cuda)
                    losses.append(loss.item())

                    if loss.item() != 0:
                        loss.backward()
                        optimizer.step()
            else:
                loss = 0
                for k, (sample, click, propensity) in enumerate(BatchIterator(train_set, batch_size, enable_cuda, sparse, device)):
                    if k % batch_size == 0:
                        if loss != 0:
                            loss.backward()
                            optimizer.step()
                            losses.append(loss.item() / batch_size)
                        loss = 0
                    optimizer.zero_grad()
                    output = model(sample)
                    loss += calc_loss(output, click, propensity, lamb, 0, enable_cuda)
                    
        epoch_losses.append(sum(losses) / len(losses))
        logging.info("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))

        run_test_set(model, test, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device)
        torch.save(model.state_dict(), save_model_path + '_{}.pt'.format(i))
        torch.save(optimizer.state_dict(), save_model_path + '_optimizer_{}.pt'.format(i))

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