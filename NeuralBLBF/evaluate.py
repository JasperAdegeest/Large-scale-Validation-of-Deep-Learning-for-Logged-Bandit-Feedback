import torch
import logging

import numpy as np
from tqdm import tqdm 
from NeuralBLBF.data import BatchIterator, get_start_stop_idx, CriteoDataset


def run_test_set(model, test_filename, batch_size, enable_cuda, sparse, feature_dict, stop_idx, step_size, save, device):
    model.eval()
    with torch.no_grad():
        R = 0
        C = 0
        N = 0

        for i in range(0, stop_idx, step_size):
            logging.info("Loading testing {} to {} out of {}.".format(i, i+step_size, stop_idx))
            test_set = CriteoDataset(test_filename, feature_dict, i+step_size, i, sparse, save)
            for j, (sample, click, propensity) in enumerate(BatchIterator(test_set, batch_size, enable_cuda, sparse, device)):
                output = model(sample)
                rectified_label = click.eq(0).float()
                N += torch.sum(click.eq(1) * 10 + click.eq(0)).float()
                R += torch.sum(rectified_label * (output[:, 0, 0] / propensity))
                C += torch.sum(output[:, 0, 0] / propensity)

        R = (R / N) * 10**4
        C = C / N
        R_div_C = R / C

        logging.info("Test Results: R x 10^4: {:.4f}\t C: {:.4f}\t (R x 10^4) / C: {:.4f}".format(R, C, R_div_C))
