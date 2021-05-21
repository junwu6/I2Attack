from __future__ import print_function
import argparse
import data_loader
import torch
import os
import copy
import math
from utils import *
from models.I2Attack import run_I2Attack
from models.MDD import MDDModel
from models.DANN import DANNModel
from models.DAN import DANModel

# Command setting
parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('-model', type=str, default='MDD', help='model name')
parser.add_argument('-mode', type=str, default='poison', help='poison|clean')
parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-test_batch_size', type=int, default=500, help='test batch size')
parser.add_argument('-cuda', type=int, default=0, help='cuda id')
parser.add_argument('-root_dir', type=str, default='E:/Codes/data/office/')
parser.add_argument('-source', type=str, default='webcam_list.txt')
parser.add_argument('-target', type=str, default='amazon_list.txt')
parser.add_argument('-epochs', type=int, default=2000)
parser.add_argument('-num_classes', type=int, default=31)  # 31 for office; 65 for office-home; 12 for image-clef, visda
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-moment', type=float, default=0.9)
parser.add_argument('-l2_decay', type=float, default=5e-4)
args = parser.parse_args()


def train(src_data, tgt_data, tgt_test_data, device):
    model_name = args.model
    model = None
    print(model_name)
    if model_name == 'MDD':
        model = MDDModel(num_classes=args.num_classes).to(device)
    elif model_name == 'DAN':
        model = DANModel(num_classes=args.num_classes).to(device)
    elif model_name == 'DANN':
        model = DANNModel(num_classes=args.num_classes).to(device)
    optimizer = get_optimizer(model, args)

    if args.mode == 'poison':
        src_data = run_I2Attack(copy.deepcopy(model), src_data, tgt_data, device, eps=0.1, step_size=0.01, attack_epochs=25)
        print("Poisoning source is done")
    src_generator = batch_generator(src_data, batch_size=args.batch_size)
    tgt_generator = batch_generator(tgt_data, batch_size=args.batch_size)

    for i in range(args.epochs):
        model.train()
        learning_rate = args.lr / math.pow((1 + 10 * i / args.epochs), 0.75)
        adjust_learning_rate(optimizer, learning_rate)

        sinputs, slabels = next(src_generator)
        tinputs, _ = next(tgt_generator)
        sinputs = torch.tensor(sinputs, requires_grad=False, dtype=torch.float).to(device)
        slabels = torch.tensor(slabels, requires_grad=False, dtype=torch.long).to(device)
        tinputs = torch.tensor(tinputs, requires_grad=False, dtype=torch.float).to(device)

        optimizer.zero_grad()
        alpha = 2 / (1 + math.exp(-10 * i / args.epochs)) - 1
        loss = model(sinputs, slabels, tinputs, alpha)
        loss.backward()
        optimizer.step()
        # print('Epoch: [{:02d}/{:02d}], loss: {:.6f}'.format(i + 1, args.epochs, loss.item()))

        if (i + 1) % 200 == 0:
            model.eval()
            test_acc = 0.
            with torch.no_grad():
                test_len = tgt_test_data['X'].shape[0] // args.test_batch_size
                for j in range(test_len):
                    outputs = model.inference(torch.tensor(tgt_test_data['X'][args.test_batch_size * j:args.test_batch_size * (j + 1)], requires_grad=False).to(device))
                    preds = torch.max(outputs, 1)[1]
                    test_acc += torch.sum(preds == torch.tensor(tgt_test_data['Y'][args.test_batch_size * j:args.test_batch_size * (j + 1)], requires_grad=False, dtype=torch.long).to(device))
                if test_len * args.test_batch_size < tgt_test_data['X'].shape[0]:
                    outputs = model.inference(torch.tensor(tgt_test_data['X'][test_len * args.test_batch_size:], requires_grad=False).to(device))
                    preds = torch.max(outputs, 1)[1]
                    test_acc += torch.sum(preds == torch.tensor(tgt_test_data['Y'][test_len * args.test_batch_size:], requires_grad=False, dtype=torch.long).to(device))
            test_acc = test_acc.double() / tgt_test_data['X'].shape[0]

            with torch.no_grad():
                train_acc = 0.
                train_len = src_data['X'].shape[0] // args.test_batch_size
                for j in range(train_len):
                    outputs = model.inference(torch.tensor(src_data['X'][args.test_batch_size * j:args.test_batch_size * (j + 1)], requires_grad=False).to(device))
                    preds = torch.max(outputs, 1)[1]
                    train_acc += torch.sum(preds == torch.tensor(src_data['Y'][args.test_batch_size * j:args.test_batch_size * (j + 1)], requires_grad=False, dtype=torch.long).to(device))
                if train_len * args.test_batch_size < src_data['X'].shape[0]:
                    outputs = model.inference(torch.tensor(src_data['X'][train_len * args.test_batch_size:], requires_grad=False).to(device))
                    preds = torch.max(outputs, 1)[1]
                    train_acc += torch.sum(preds == torch.tensor(src_data['Y'][train_len * args.test_batch_size:], requires_grad=False, dtype=torch.long).to(device))
            train_acc = train_acc.double() / src_data['X'].shape[0]

            with torch.no_grad():
                discrepancy = 0.
                num_examples = 32
                num_batchs = min(src_data['X'].shape[0], tgt_data['X'].shape[0]) // num_examples
                for j in range(num_batchs):
                    s_val_inputs = torch.tensor(src_data['X'][num_examples * j: num_examples * (j + 1)], requires_grad=False).to(device)
                    t_val_inputs = torch.tensor(tgt_data['X'][num_examples * j: num_examples * (j + 1)], requires_grad=False).to(device)
                    discrepancy += model.get_discrepancy(s_val_inputs, t_val_inputs)
                discrepancy = discrepancy / num_batchs
            print('Epoch: [{:02d}/{:02d}], loss: {:.6f}, train acc: {:.4f}, discrepancy: {:.4f}, test acc: {:.4f}'.format(i + 1, args.epochs, loss.item(), train_acc, discrepancy, test_acc))

    return test_acc


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(23)

    if not os.path.isfile("data_preprocessed/{}.pkl".format(args.source[:-9])):
        raw_src_loader = data_loader.load_training(args.root_dir, args.source, args.batch_size)
        save_data(raw_src_loader, name=args.source[:-9])
    src_data = pickle.load(open("data_preprocessed/{}.pkl".format(args.source[:-9]), "rb"))

    if not os.path.isfile("data_preprocessed/{}.pkl".format(args.target[:-9])):
        raw_tgt_loader = data_loader.load_training(args.root_dir, args.target, args.batch_size)
        save_data(raw_tgt_loader, name=args.target[:-9])
    tgt_data = pickle.load(open("data_preprocessed/{}.pkl".format(args.target[:-9]), "rb"))

    if not os.path.isfile("data_preprocessed/{}_test.pkl".format(args.target[:-9])):
        raw_tgt_loader = data_loader.load_testing(args.root_dir, args.target, args.batch_size)
        save_data(raw_tgt_loader, name=args.target[:-9]+'_test')
    tgt_test_data = pickle.load(open("data_preprocessed/{}_test.pkl".format(args.target[:-9]), "rb"))

    test_acc = train(src_data, tgt_data, tgt_test_data, device)
