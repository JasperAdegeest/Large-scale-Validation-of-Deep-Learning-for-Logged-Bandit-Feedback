import torch
import logging

import numpy as np
from tqdm import tqdm 

from NeuralBLBF.data import BatchIterator


def run_test_set(model, test_set, batch_size, enable_cuda, hasher):
    model.eval()
    with torch.no_grad():
        R = 0
        C = 0
        N = 0

        for sample, click, propensity in BatchIterator(test_set, batch_size, enable_cuda, hasher):
            output = model(sample)
            rectified_label = click.eq(0).float()
            N += torch.sum(click.eq(1) * 10 + click.eq(0)).float()
            R += torch.sum(rectified_label * (output[:, 0, 0] / propensity))
            C += torch.sum(output[:, 0, 0] / propensity)

        R = (R / N) * 10**4
        C = C / N
        R_div_C = R / C

        logging.info("\nTest results:")
        logging.info("R x 10^4: {}\t C: {}\t (R x 10^4) / C: {}\n".format(R, C, R_div_C))
