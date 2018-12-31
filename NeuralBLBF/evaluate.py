import torch
import logging

import numpy as np
from tqdm import tqdm 
from NeuralBLBF.data import BatchIterator, get_start_stop_idx, CriteoDataset


def run_test_set(model, test_filename, batch_size, enable_cuda, sparse,
                 feature_dict, stop_idx, step_size, save, device, **kwargs):
    """
        Evaluates the model based on a test set. The following evaluation metrics will be calculated:
            - R
            - C
            - R / C
        Disclosure: The calculation of the metrics is inspired by the Scripts/scorer.py code provided by Criteo
    """

    model.eval()
    with torch.no_grad(), open("propensities_lp.txt", 'w') as f1, open("propensities_np.txt", 'w') as f2:
        modifiedDenomList = []
        numerator = []
        denominator = []

        # Extract the Numerator, Denominator and modifiedDenominator information out of the test set
        for i in range(0, stop_idx, step_size):
            logging.info("Loading testing {} to {} out of {} of test set: {}.".format(i, i+step_size, stop_idx, test_filename))
            test_set = CriteoDataset(test_filename, feature_dict, i+step_size, i, sparse, save)
            for j, (sample, click, propensity) in enumerate(BatchIterator(test_set, batch_size, enable_cuda, sparse, device)):
                output = model(sample, 0.0)

                rectified_label = click.eq(0).float()
                a = click.eq(1) * 10 + click.eq(0)
                modifiedDenomList.extend(a.cpu().numpy())

                b = rectified_label * (output[:, 0, 0] / propensity)

                numerator.extend(b.cpu().numpy())

                c = output[:, 0, 0] / propensity

                denominator.extend(c.cpu().numpy())
                output = output.squeeze(2)

                # Save propensities to text file for later analysis
                for c, p in zip(list(click.cpu().numpy()), list(output[:, 0].cpu().numpy())):
                    f1.write("{}\t{}\n".format(c, p))
                sampling = torch.multinomial(output, 1, replacement=False).cpu().numpy()
                for c, index, prop in zip(click.cpu().numpy(), sampling, output.cpu().numpy()):
                    f2.write("{}\n".format(prop[index[0]]))

        modifiedDenom = sum(modifiedDenomList)
        power = 10**4
        maxInstances = len(modifiedDenomList)
        scaleFactor = np.sqrt(maxInstances) / modifiedDenom

        # Calculate R values
        R = (np.sum(numerator) / modifiedDenom)
        R_std = 2.58 * np.std(numerator) * scaleFactor  # 99% CI

        # Calculate C values
        C = np.sum(denominator) / modifiedDenom
        C_std = 2.58 * np.std(denominator) * scaleFactor  # 99% CI

        # Calculate R/C values
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
                     .format(R*power, R_std*power, C, C_std, R_div_C*power, R_div_C_std*power))
