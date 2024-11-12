import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim



def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    

def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args

def get_optimizer(tree, args: argparse.Namespace) -> torch.optim.Optimizer:
    """
    Construct the optimizer as dictated by the parsed arguments
    :param tree: The tree that should be optimized
    :param args: Parsed arguments containing hyperparameters. The '--optimizer' argument specifies which type of
                 optimizer will be used. Optimizer specific arguments (such as learning rate and momentum) can be passed
                 this way as well
    :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen, and parameter set of the net that will be trained
    """

    optim_type = args.optimizer
    #create parameter groups
    params_to_freeze = []
    params_to_train = []

    dist_params = []
    for name,param in tree.named_parameters():
        if 'dist_params' in name:
            dist_params.append(param)
    # set up optimizer
    if 'resnet50_inat' in args.net or ('resnet50' in args.net and args.dataset=='CARS'):  #to reproduce experimental results
        # freeze resnet50 except last convolutional layer
        for name,param in tree._net.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)
   
        if optim_type == 'SGD':
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum}, 
                {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
                {"params": tree.prototype_layer.parameters(), "lr": args.lr,"weight_decay_rate": 0,"momentum": 0}]
            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})
        else:
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay}, 
                {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": tree.prototype_layer.parameters(), "lr": args.lr,"weight_decay_rate": 0}]
            
            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})
    
    else: #other network architectures
        for name,param in tree._net.named_parameters():
            params_to_freeze.append(param)
        paramlist = [
            {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay}, 
            {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": tree.prototype_layer.parameters(), "lr": args.lr,"weight_decay_rate": 0}]
        if args.disable_derivative_free_leaf_optim:
            paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})
    
    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist,
                               lr=args.lr,
                               momentum=args.momentum), params_to_freeze, params_to_train
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist,lr=args.lr,eps=1e-07), params_to_freeze, params_to_train
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist,lr=args.lr,eps=1e-07, weight_decay=args.weight_decay), params_to_freeze, params_to_train

    raise Exception('Unknown optimizer argument given!')


