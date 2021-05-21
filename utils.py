from __future__ import print_function
import numpy as np
import torch.optim as optim
import pickle


def get_optimizer(model, args):
    learning_rate = args.lr
    param_group = []
    for k, v in model.named_parameters():
        if k.__contains__('base_network'):
            param_group += [{'name': k, 'params': v, 'lr': learning_rate / 10}]
        else:
            param_group += [{'name': k, 'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group, lr=learning_rate, momentum=args.moment, weight_decay=args.l2_decay)
    return optimizer


def adjust_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        name = param_group['name']
        if name.__contains__('base_network'):
            param_group['lr'] = learning_rate / 10
        else:
            param_group['lr'] = learning_rate


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(raw_data, batch_size, shuffle=True):
    data = [raw_data['X'], raw_data['Y']]
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[0][start:end], data[1][start:end]


def save_data(dataloader, name):
    X, Y = [], []
    for data, target in dataloader:
        X.append(data.detach().cpu().numpy())
        Y.append(target.detach().cpu().numpy())
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    print(X.shape, Y.shape)

    data = {}
    data['X'] = X
    data['Y'] = Y

    with open("data_preprocessed/{}.pkl".format(name), "wb") as pkl_file:
        pickle.dump(data, pkl_file)
