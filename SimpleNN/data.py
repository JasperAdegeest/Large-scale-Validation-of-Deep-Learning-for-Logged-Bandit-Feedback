import argparse
import torch
from SimpleNN.config import Config

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset


class Sample():
    """Sample representing banner with one slot."""
    def __init__(self):
        self.config = Config()

        # Start with empty product list
        self.products = []

        # All should be initialized by functions of the sample
        self.summary = None
        self.product_vecs = None
        self.click = None
        self.propensity = None

    def done(self):
        # Summary vec will be reused for all product vecs
        summary = self.summary.split("|")[-1]

        # Extract click and propensity from first product
        product_showed = self.products[0]
        [score, features] = product_showed.split("|")
        [_, self.click, self.propensity] = score.split(":")
        self.click = float(self.click)
        self.propensity = float(self.propensity)
        first_product_vector = self.features_to_vector(summary + ' ' + features)

        # Extract feature vecs for every other product as well
        self.product_vecs = [first_product_vector]
        for p in self.products[1:]:
            vec = self.features_to_vector(summary + ' ' + p.split("|")[-1])
            self.product_vecs.append(vec)

    def features_to_vector(self, features):
        # For now, numerical and categorical features are cast to numbers
        vector = [0] * 35
        for feature in features.split():
            if ":" in feature and not "_" in feature:
                [feature_name, value] = feature.split(":")
                vector[int(feature_name)-1] = int(value)
            if "_" in feature:
                [feature_name, value] = feature.split("_")
                if ":" in value: value = value.split(":")[0]
                category_index = self.config.get_category_index(int(feature_name), int(value))
                if category_index is not None:
                    vector[int(feature_name)-1] = category_index
        return vector

    def __str__(self):
        to_string = "Summary: {}\n".format(self.summary)
        to_string += "Products\n"
        for p in self.products:
            to_string += "---- {}\n".format(p)
        return to_string


class BatchIterator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for sample in self.dataset:
            yield torch.Tensor(sample.product_vecs).float(), sample.click, sample.propensity


class CriteoDataset(Dataset):
    """Criteo dataset."""

    def __init__(self, filename, stop_idx=10000000):
        """
        Args:
            filename (string): Path to the criteo dataset filename.
            stop_idx (int): only process this many lines from the file.
        """
        self.samples = []
        self.load(filename, stop_idx)

    def load(self, filename, stop_idx):
        with open(filename) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i >= stop_idx: break

                # Line break
                if not line:
                    continue
                # Start of new sample
                elif "shared" in line:
                    if i != 0:
                        sample.done()
                        self.samples.append(sample)
                    sample = Sample()
                    sample.summary = line
                # Product line
                else:
                    sample.products.append(line)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', required=True)
    args = parser.parse_args()

    train = CriteoDataset(args.train, 500000)
