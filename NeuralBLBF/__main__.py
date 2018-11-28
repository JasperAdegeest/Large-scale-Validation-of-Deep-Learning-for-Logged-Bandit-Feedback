import argparse
import torch
import logging

import numpy as np
from tqdm import tqdm 
from NeuralBLBF.train import train
from NeuralBLBF.model import EmbedFFNN, HashFFNN
from NeuralBLBF.data import CriteoDataset, BatchIterator
from sklearn.feature_extraction import FeatureHasher


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', default='data/vw_compressed_train')
    parser.add_argument('--test', default='data/vw_compressed_test')
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stop_idx', type=int, default=500000)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--feature_dict', type=str, default='data/feature_to_keys.json')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--learning_rate', default=0.0001)

    # If hashing is used the model needs to be changed
    parser.add_argument('--hashing', action='store_true')
    parser.add_argument('--model', default="EmbedFFN", choices=["EmbedFFN", "HashFNN"])
    parser.add_argument('--n_features', default=2**12, type=int)
    args = parser.parse_args()

    if not args.hashing:
        hasher = None
        logging.info("Loading training dataset.")
        train_set = CriteoDataset(args.train, args.feature_dict, args.stop_idx, args.start_idx)
        logging.info("Finished loading training dataset, loading testing dataset now.")
        test_set = CriteoDataset(args.test, args.feature_dict, args.stop_idx, args.start_idx)
        logging.info("Finished loading testing datset, initialising model now.")
        model = EmbedFFNN(args.embedding_dim, args.hidden_dim, train_set.feature_dict, args.cuda)
    else:
        hasher = FeatureHasher(n_features=args.n_features, input_type="dict")
        logging.info("Loading training dataset.")
        train_set = CriteoDataset(args.train, args.feature_dict, args.stop_idx, args.start_idx, hashing=True)
        logging.info("Finished loading training dataset, loading testing dataset now.")
        test_set = CriteoDataset(args.test, args.feature_dict, 30000000+args.stop_idx, 30000000, hashing=True)
        logging.info("Finished loading testing datset, initialising model now.")
        model = HashFFNN(args.n_features)        

    if args.cuda and torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model, optimizer, train_set, test_set, args.batch_size, args.cuda, args.epochs, args.lamb, hasher)
