import torch
import logging

import numpy as np
from tqdm import tqdm 
from NeuralBLBF.data import BatchIterator, get_start_stop_idx, CriteoDataset


def run_test_set(model, test_filename, batch_size, enable_cuda, sparse,
                 feature_dict, stop_idx, step_size, save, device, **kwargs):
    model.eval()
    with torch.no_grad(), open("propensities_lp.txt", 'w') as f1, open("propensities_np.txt", 'w') as f2:
        R = 0
        C = 0
        N = 0
        N_list = []
        R_list = []
        C_list = []

        for i in range(0, stop_idx, step_size):
            logging.info("Loading testing {} to {} out of {} of test set: {}.".format(i, i+step_size, stop_idx, test_filename))
            test_set = CriteoDataset(test_filename, feature_dict, i+step_size, i, sparse, save)
            for j, (sample, click, propensity) in enumerate(BatchIterator(test_set, batch_size, enable_cuda, sparse, device)):
                output = model(sample, 0.0)
                rectified_label = click.eq(0).float()
                a = click.eq(1) * 10 + click.eq(0)
                N_list.extend(a.cpu().numpy())

                b = rectified_label * (output[:, 0, 0] / propensity)
                R_list.extend(b.cpu().numpy())

                c = output[:, 0, 0] / propensity
                C_list.extend(c.cpu().numpy())
                output = output.squeeze(2)


                for c, p in zip(list(click.cpu().numpy()), list(output[:, 0].cpu().numpy())):
                    f1.write("{}\t{}\n".format(c, p))

                sampling = torch.multinomial(output, 1, replacement=False).cpu().numpy()
                for c, index, prop in zip(click.cpu().numpy(), sampling, output.cpu().numpy()):
                    f2.write("{}\n".format(prop[index[0]]))


        modifiedDenom = sum(N_list)
        power = 10**4
        numerator = R_list
        denominator = C_list
        maxInstances = len(R_list)
        scaleFactor = np.sqrt(maxInstances) / modifiedDenom

        R = (np.sum(numerator) / modifiedDenom) * power
        R_std = 2.58 * np.std(numerator) * scaleFactor  # 99% CI
        C = np.sum(denominator) / modifiedDenom
        C_std = 2.58 * np.std(denominator) * scaleFactor  # 99% CI
        R_div_C = R / C

        normalizer = C * modifiedDenom

        # See Art Owen, Monte Carlo, Chapter 9, Section 9.2, Page 9
        # Delta Method to compute an approximate CI for SN-IPS
        Var = np.sum(np.square(numerator) + \
                        np.square(denominator) * R_div_C * R_div_C - \
                        2 * R_div_C * np.multiply(numerator, denominator), dtype=np.longdouble) / (
                          normalizer * normalizer)
        R_div_C_std = 2.58 * np.sqrt(Var) / np.sqrt(maxInstances)  # 99% CI

        logging.info("Test Results: R x 10^4: {:.4f}+/-{:.3f}\t C: {:.4f}+/-{:.3f}\t (R x 10^4) / C: {:.4f}+/-{:.3f}"
                     .format(R, R_std, C, C_std, R_div_C, R_div_C_std))
