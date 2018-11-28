import argparse
import torch
import os
import random
import json
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict


class Sample():
    """Sample representing banner with one slot."""
    def __init__(self, feature_dict):

        # Start with empty product list
        self.products = []
        self.feature_dict = feature_dict

        # All should be initialized by functions of the sample
        self.summary = None
        self.product_vecs = None
        self.click = None
        self.propensity = None

    def done(self, hashing):
        # Summary vec will be reused for all product vecs
        summary = self.summary.split("|")[-1]

        # Extract click and propensity from first product
        product_showed = self.products[0]
        [score, features] = product_showed.split("|")
        [_, self.click, self.propensity] = score.split(":")
        self.click = round(float(self.click))
        self.propensity = float(self.propensity)
        if not hashing:
            first_product_vector = self.features_to_vector(summary + ' ' + features)
        else:
            first_product_vector = self.features_to_vector_hashed(summary + ' ' + features)            

        # Extract feature vecs for every other product as well
        self.product_vecs = [first_product_vector]
        for p in self.products[1:]:
            if not hashing:
                vec = self.features_to_vector(summary + ' ' + p.split("|")[-1])
            else:
                vec = self.features_to_vector_hashed(summary + ' ' + p.split("|")[-1])
            self.product_vecs.append(vec)

    def features_to_vector(self, features):
        vector = [0] * 35
        for feature in features.split():
            if ":" in feature and not "_" in feature:
                [feature_name, value] = feature.split(":")
                vector[int(feature_name)-1] = int(value)
            if "_" in feature:
                [feature_name, value] = feature.split("_")
                if ":" in value: value = value.split(":")[0]
                category_index = self.get_category_index(int(feature_name), int(value))
                if category_index is not None:
                    vector[int(feature_name)-1] = category_index
        return vector

    def features_to_vector_hashed(self, features):
        feature_dict = dict()
        for feature in features.split():
            if ":" in feature and not "_" in feature:
                [feature_name, value] = feature.split(":")
                feature_dict[feature_name] = int(value)
            if "_" in feature:
                [feature_name, value] = feature.split("_")
                if ":" in value: value = value.split(":")[0]
                feature_dict[feature_name] = value
        return feature_dict

    def __str__(self):
        to_string = "Summary: {}\n".format(self.summary)
        to_string += "Products\n"
        for p in self.products:
            to_string += "---- {}\n".format(p)
        return to_string

    def get_category_index(self, feature, category):
        if str(category) in self.feature_dict[str(feature)]:
            return self.feature_dict[str(feature)][str(category)]
        else:
            return None


class BatchIterator():
    def __init__(self, dataset, batch_size, enable_cuda, hasher=None):
        self.dataset = dataset
        self.sorted_per_pool_size = defaultdict(list)
        for s in self.dataset:
            self.sorted_per_pool_size[len(s.products)].append(s)
        self.sorted_per_pool_size = dict(self.sorted_per_pool_size)
        self.batch_size = batch_size
        self.enable_cuda = enable_cuda
        self.hasher = hasher

    def __iter__(self):
        for pool_size in self.sorted_per_pool_size:
            data = self.sorted_per_pool_size[pool_size]
            random.shuffle(data)
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                products = [sample.product_vecs for sample in batch]
                if self.hasher is not None:
                    products = self.hash(products)
                products = torch.FloatTensor(products)
                clicks = torch.FloatTensor([sample.click for sample in batch])
                propensities = torch.FloatTensor([sample.propensity for sample in batch])
                if self.enable_cuda:
                    products = products.cuda()
                    clicks = clicks.cuda()
                    propensities = propensities.cuda()
                yield products, clicks, propensities

    def hash(self, batch):
        for i, product_vecs in enumerate(batch):
            product_vecs = self.hasher.transform(product_vecs).toarray()
            batch[i] = product_vecs
        return batch


class CriteoDataset(Dataset):
    """Criteo dataset."""

    def __init__(self, filename, features_config, stop_idx=10000000, start_idx=0,
                 hashing=False):
        """
        Args:
            filename (string): Path to the criteo dataset filename.
            stop_idx (int): only process this many lines from the file.
        """

        self.samples = []
        self.hashing = hashing
        with open(features_config) as f:
            self.feature_dict = json.load(f)
        self.load(filename, stop_idx, start_idx)


    def load(self, filename, stop_idx, start_idx):
        pickle_file = '{}_{}.pickle'.format(filename, stop_idx)
        sample = None

        if os.path.exists(pickle_file):
            self.samples = pickle.load(open(pickle_file, "rb"))
        else:
            with open(filename) as f:
                for i, line in tqdm(enumerate(f)):
                    line = line.strip()
                    # Start after certain index
                    if i < start_idx: continue
                    # Stop before certain index
                    if stop_idx != -1 and i >= stop_idx: break
                    # Line break
                    if not line: continue

                    # Start of new sample
                    elif "shared" in line:
                        if sample is not None:
                            sample.done(self.hashing)
                            self.samples.append(sample)
                        sample = Sample(feature_dict=self.feature_dict)
                        sample.summary = line
                    # Product line
                    else:
                        if sample is not None:
                            sample.products.append(line)
                # Save for usage later
                pickle.dump(self.samples, open(pickle_file, 'wb'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
