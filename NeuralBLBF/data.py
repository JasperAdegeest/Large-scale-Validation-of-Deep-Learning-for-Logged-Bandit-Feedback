import argparse
import torch
import os
import random
import json
import pickle
import logging
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset
from collections import defaultdict


def get_start_stop_idx(filename):
    for part in filename.split("_"):
        if "-" in part:
            [start_idx, stop_idx] = part.split("-")
    return start_idx, stop_idx


class Sample():
    """Sample representing banner with one slot."""
    def __init__(self):

        # Start with empty product list
        self.products = []

        # All should be initialized by functions of the sample
        self.summary = None
        self.click = None
        self.propensity = None

    def done(self, sparse, feature_dict=None):
        # Summary vec will be reused for all product vecs
        summary = self.summary.split("|")[-1]

        # Extract click and propensity from first product
        product_showed = self.products[0]
        [score, features] = product_showed.split("|")
        [_, self.click, self.propensity] = score.split(":")
        self.click = round(float(self.click))
        self.propensity = float(self.propensity)

        # Extract feature vecs for every other product as well
        if not sparse:
            first_product_vector = self.features_to_vector(summary + ' ' + features, feature_dict)
            self.products[0] = first_product_vector
            for i, p in enumerate(self.products[1:]):
                vec = self.features_to_vector(summary + ' ' + p.split("|")[-1], feature_dict)
                self.products[i+1] = vec
        else:
            indices_1, indices_2, values = self.features_to_vector_sparse(summary + ' ' + features, feature_dict, 0)
            for i, p in enumerate(self.products[1:]):
                features = summary + ' ' + p.split("|")[-1]
                i_1, i_2, v = self.features_to_vector_sparse(features, feature_dict, i+1)
                indices_1.extend(i_1)
                indices_2.extend(i_2)
                values.extend(v)
            indices = torch.LongTensor([indices_1, indices_2])
            values = torch.FloatTensor(values)
            self.products = Variable(torch.sparse.FloatTensor(
                indices, values, 
                (len(self.products), len(feature_dict))
            ))


    def features_to_vector(self, features, feature_dict):
        vector = [0] * 35
        for feature in features.split():
            if ":" in feature and not "_" in feature:
                [feature_name, value] = feature.split(":")
                vector[int(feature_name)-1] = int(value)
            if "_" in feature:
                [feature_name, value] = feature.split("_")
                if ":" in value: value = value.split(":")[0]
                category_index = self.get_category_index(int(feature_name), int(value), feature_dict)
                if category_index is not None:
                    vector[int(feature_name)-1] = category_index
        return vector

    def features_to_vector_sparse(self, features, feature_dict, index):
        indices = []
        values = []
        for feature in features.split():
            if ":" in feature:
                [feature, value] = feature.split(":")
                if feature in feature_dict:
                    indices.append(feature_dict[feature])
                    values.append(int(value))
            else:
                if feature in feature_dict:
                    indices.append(feature_dict[feature])
                    values.append(1)
        return [index] * len(indices), indices, values

    def __str__(self):
        to_string = "Summary: {}\n".format(self.summary)
        to_string += "Products\n"
        for p in self.products:
            to_string += "---- {}\n".format(p)
        return to_string

    def get_category_index(self, feature, category, feature_dict):
        if str(category) in feature_dict[str(feature)]:
            return feature_dict[str(feature)][str(category)]
        else:
            return None


class BatchIterator():
    def __init__(self, dataset, batch_size, enable_cuda, sparse=False, device=None):
        self.dataset = dataset
        self.sorted_per_pool_size = defaultdict(list)
        for s in self.dataset:
            self.sorted_per_pool_size[len(s.products)].append(s)
        self.sorted_per_pool_size = dict(self.sorted_per_pool_size)
        self.batch_size = batch_size
        self.enable_cuda = enable_cuda
        self.sparse = sparse
        self.device = device

    def __iter__(self):
        if self.sparse:
            for sample in self.dataset:
                products = sample.products
                clicks = torch.FloatTensor([sample.click])
                propensities = torch.FloatTensor([sample.propensity])
                if self.enable_cuda:
                    products = products.to(self.device)
                    clicks = clicks.to(self.device)
                    propensities = propensities.to(self.device)
                yield products, clicks, propensities
        else:
            for pool_size in self.sorted_per_pool_size:
                data = self.sorted_per_pool_size[pool_size]
                random.shuffle(data)
                for i in range(0, len(data), self.batch_size):
                    batch = data[i:i+self.batch_size]
                    products = [sample.products for sample in batch]
                    products = torch.FloatTensor(products)
                    clicks = torch.FloatTensor([sample.click for sample in batch])
                    propensities = torch.FloatTensor([sample.propensity for sample in batch])
                    if self.enable_cuda:
                        products = products.to(self.device)
                        clicks = clicks.to(self.device)
                        propensities = propensities.to(self.device)
                    yield products, clicks, propensities


class CriteoDataset(Dataset):
    """Criteo dataset."""

    def __init__(self, filename, features_dict, stop_idx=10000000, start_idx=0,
                 sparse=False, save=False):
        """
        Args:
            filename (string): Path to the criteo dataset filename.
            stop_idx (int): only process this many lines from the file.
        """

        self.samples = []
        self.save = save
        self.sparse = sparse
        self.feature_dict = features_dict
        self.load(filename, stop_idx, start_idx, sparse)


    def load(self, filename, stop_idx, start_idx, sparse):
        if sparse:
            pickle_file = '{}_{}-{}_sparse.pickle'.format(filename, start_idx, stop_idx)
        else:
            pickle_file = '{}_{}-{}.pickle'.format(filename, start_idx, stop_idx)

        sample = None
        if os.path.exists(pickle_file):
            self.samples = pickle.load(open(pickle_file, "rb"))
        else:
            with open(filename) as f:
                for i, line in enumerate(f):
                    #if i % 50000 == 0: print(i)
                    line = line.strip()
                    # Start after certain index
                    if start_idx != -1 and i < start_idx: continue
                    # Stop before certain index
                    if stop_idx != -1 and i >= stop_idx: break
                    # Line break
                    if not line: continue

                    # Start of new sample
                    elif "shared" in line:
                        if sample is not None:
                            sample.done(self.sparse, self.feature_dict)
                            self.samples.append(sample)
                        sample = Sample()
                        sample.summary = line
                    # Product line
                    else:
                        if sample is not None:
                            sample.products.append(line)

                # Save for usage later
                if self.save: pickle.dump(self.samples, open(pickle_file, 'wb'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', default='data/vw_compressed_train')
    parser.add_argument('--stop_idx', type=int, default=500000)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--feature_dict', type=str, default='data/feature_to_keys.json')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sparse', action='store_true')
    args = parser.parse_args()

    train_set = CriteoDataset(args.data, args.feature_dict, args.stop_idx, args.start_idx, args.sparse, args.save)