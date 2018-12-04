import argparse
import torch
import logging
import json
import os
import numpy as np

from NeuralBLBF.train import train
from NeuralBLBF.model import TinyEmbedFFNN, SmallEmbedFFNN, HashFFNN, LargeEmbedFFNN, CrossNetwork


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', default='data/vw_compressed_train')
    parser.add_argument('--valid', default='data/vw_compressed_validate')
    parser.add_argument('--test', default='data/vw_compressed_train')
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stop_idx', type=int, default=1500)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--feature_dict_name', type=str, default='data/features_to_keys.json')
    parser.add_argument('--enable_cuda', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--step_size', type=int, default=100000)
    parser.add_argument('--learning_rate', default=0.00005)
    parser.add_argument('--save_model_path', type=str, default='data/models')
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--optimizer_path', type=str, default=None)
    parser.add_argument('--prop_dropout', action='store_true', help="Use propensity dropout [https://arxiv.org/pdf/1706.05966.pdf]")

    # If sparse is used the model needs to be changed
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--model_type', default="TinyEmbedFFNN", choices=["TinyEmbedFFNN", "SmallEmbedFFNN", "HashFFNN", "LargeEmbedFFNN", "CrossNetwork"])
    args = vars(parser.parse_args())

    if args['enable_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda', args['device_id'])
    else:
        device = None

    logging.info("Parameters:")
    for k, v in args.items():
        logging.info("  %12s : %s" % (k, v))

    # Load dict mapping features to keys
    with open(args['feature_dict_name']) as f: feature_dict = json.load(f)
    if not os.path.exists(args['save_model_path']):
        os.mkdir(args['save_model_path'])

    args['save_model_path'] = '{}/{}_{}'.format(
        args['save_model_path'], args['model_type'], args['embedding_dim']
    )

    # Initialize neural architecture and optimizer to use
    if args['model_type'] == "TinyEmbedFFNN" and not args['sparse']:
        model = TinyEmbedFFNN(feature_dict, device, **args)
    elif args['model_type'] == "HashFFNN" or args['sparse']:
        model = HashFFNN(len(feature_dict))
    elif args['model_type'] == "LargeEmbedFFNN":
        model = LargeEmbedFFNN(feature_dict, device, **args)
    elif args['model_type'] == "SmallEmbedFFNN":
        model = SmallEmbedFFNN(feature_dict, device, **args)
    elif args['model_type'] == "CrossNetwork":
        model = CrossNetwork(feature_dict, device, **args)
    else:
        raise NotImplementedError()

    n_params = sum([np.prod(par.size()) for par in model.parameters() if par.requires_grad])
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    if args['model_path'] is not None and args['optimizer_path'] is not None:
        checkpoint = torch.load(args['model_path'])
        model.load_state_dict(checkpoint)
        checkpoint = torch.load(args['optimizer_path'])
        optimizer.load_state_dict(checkpoint)

    if args["enable_cuda"] and torch.cuda.is_available(): model.to(device)
    logging.info("Initialized model and optimizer. Number of parameters: {}".format(n_params))

    train(model, optimizer, feature_dict, device, **args)
