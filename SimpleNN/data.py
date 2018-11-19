import argparse
import torch

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from SimpleNN.Config import Config


class Sample():
    """Sample representing banner with one slot."""
    def __init__(self):
        self.config = Config()

        # Start with empty product list
        self.products = []

        # All should be initialized by functions of the sample
        self.summary = None
        self.summary_vec = None
        self.product_vecs = None
        self.click = None
        self.propensity = None

    def done(self):
        # Summary vec will be reused for all product vecs
        self.summary_matrix = self.features_to_matrix(self.summary.split("|")[-1])

        # Extract click and propensity from first product
        product_showed = self.products[0]
        [score, features] = product_showed.split("|")
        [_, self.click, self.propensity] = score.split(":")
        self.click = float(self.click)
        self.propensity = float(self.propensity)
        first_product_matrix = self.features_to_matrix(features)

        # Extract feature vecs for every other product as well
        self.product_mats = [first_product_matrix]
        for p in self.products[1:]:
            mat = self.features_to_matrix(p.split("|")[-1])
            mat = [max(s, p) for s, p in zip(self.summary_matrix, mat)]
            self.product_mats.append(mat)

    def __features_to_matrix(self, features):
        # For now, numerical and categorical features are cast to numbers
        matrix = []
        for i in range(34):
            if i < 2:
                matrix.append(0)
            else:
                matrix.append([0] * self.config.get_feature_size(i + 1))

        for feature in features.split():
            if ":" in feature and not "_" in feature:
                [feature_name, value] = feature.split(":")
                if int(feature_name) <= 2:
                    matrix[int(feature_name) - 1] = int(value)
                else:
                    matrix[int(feature_name) - 1][int(value) - 1] = 1
            if "_" in feature:
                [feature_name, value] = feature.split("_")
                if ":" in value: value = value.split(":")[0]
                if int(feature_name) <= 2:
                    matrix[int(feature_name) - 1] = int(value)
                else:
                    matrix[int(feature_name) - 1][int(value) - 1] = 1

        return matrix

    def features_to_matrix(self, features):
        # For now, numerical and categorical features are cast to numbers
        matrix = []
        for i in range(35):
            if i < 2:
                matrix.append(0)
            else:
                matrix.append([0])

        for feature in features.split():
            if ":" in feature and not "_" in feature:
                [feature_name, value] = feature.split(":")
                if int(feature_name) <= 2:
                    matrix[int(feature_name) - 1] = int(value)
                else:
                    if matrix[int(feature_name) - 1][0] == 0:
                        matrix[int(feature_name) - 1] = []

                    category_index = self.config.get_category_index(int(feature_name) - 1, int(value))
                    matrix[int(feature_name) - 1].append(category_index)
            if "_" in feature:
                [feature_name, value] = feature.split("_")
                if ":" in value: value = value.split(":")[0]
                if int(feature_name) <= 2:
                    matrix[int(feature_name) - 1] = int(value)
                else:
                    category_index = self.config.get_category_index(int(feature_name), int(value))
                    if category_index is not None:
                        if matrix[int(feature_name) - 1][0] == 0:
                            matrix[int(feature_name) - 1] = []

                        matrix[int(feature_name) - 1].append(category_index + 1)

        return matrix

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
            print()
            yield sample.product_mats, sample.click, sample.propensity


class CriteoDataset(Dataset):
    """Criteo dataset."""

    def __init__(self, filename, stop_idx=10000000):
        """
        Args:
            filename (string): Path to the criteo dataset filename.
            stop_idx (int): only process this many lines from the file.
        """
        self.samples = []
        #self.extract_features(filename)
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

    def extract_features(self, filename, stop_idx):
        numerical_features = set()
        categorical_features = defaultdict(set)

        with open(filename) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == stop_idx: break

                # Line break
                if not line:
                    continue
                features = line.split("|")[-1].split()
                for f in features:
                    # Numerical feature
                    if ":" in f and not "_" in f:
                        [feature_name, value] = f.split(":")
                        numerical_features.add(int(feature_name))

                    # Categorical featuer
                    if "_" in f:
                        [feature_name, value] = f.split("_")
                        if ":" in value: value = value.split(":")[0]
                        categorical_features[int(feature_name)].add(int(value))
        print(len(numerical_features))
        x = len(numerical_features)
        for f in categorical_features:
            print(f, len(categorical_features[f]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--train', required=True)
    # args = parser.parse_args()

    train = CriteoDataset('../data/vw_compressed_train', 500)
    print()
