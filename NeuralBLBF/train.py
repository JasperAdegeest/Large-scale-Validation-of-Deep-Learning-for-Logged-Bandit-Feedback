import torch
import logging
import datetime

import numpy as np
from tqdm import tqdm

from NeuralBLBF.data import CriteoDataset, BatchIterator


from NeuralBLBF.evaluate import run_test_set
from NeuralBLBF.data import BatchIterator, get_start_stop_idx, CriteoDataset


def calc_loss(output_tensor, click_tensor, propensity_tensor, lamb, gamma, enable_cuda):
    """
        Calculates the loss and returns the result
    """

    # Calculate the corrected N
    N_hat = torch.sum(click_tensor.eq(1) * 10 + click_tensor.eq(0)).float()

    # Calculate the off policy probability
    probs = (output_tensor[:, 0, 0] / propensity_tensor)

    # Calculate the final loss
    R_hat = (click_tensor - lamb) * probs
    return torch.sum(R_hat) / torch.sum(N_hat)


def train(model, optimizer, feature_dict, start_epoch, device, save_model_path, train, test,
          batch_size, enable_cuda, epochs, lamb, gamma, sparse, stop_idx, step_size,
          save, **kwargs):
    """
        Training function, initiates training/testing/saving of the model
    """
    epoch_losses = []
    logging.info("Initialized dataset")

    # Evaluate the model based on the test set and optionally the train set
    run_test_set(model, test, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device)
    if kwargs['training_eval']:
       run_test_set(model, train, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device)

    # Train the model
    for i in range(start_epoch, epochs, 1):
        logging.info("Starting epoch {}".format(i))

        losses = []
        for j in range(0, stop_idx, step_size):

            logging.info("Loading training {} to {} out of {}.".format(j, j+step_size, stop_idx))
            train_set = CriteoDataset(train, feature_dict, j+step_size, j, sparse, save)
            for k, (sample, click, propensity) in enumerate(BatchIterator(train_set, batch_size, enable_cuda, sparse, device)):
                optimizer.zero_grad()
                output = model(sample)
                loss = calc_loss(output, click, propensity, lamb, gamma, enable_cuda)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
        epoch_losses.append(sum(losses) / len(losses))
        logging.info("Finished epoch {}, avg. loss {}".format(i, epoch_losses[-1]))

        # Evaluate the model based on the test set and optionally the train set
        run_test_set(model, test, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device)
        if kwargs['training_eval']:
            run_test_set(model, train, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device)

        # Save the model
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i
        }
        logging.info("Saving after completed epoch {}".format(i))
        torch.save(state, save_model_path + 'e{}-{}.pt'.format(i, datetime.datetime.now()))
