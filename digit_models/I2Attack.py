import torch
import torch.nn as nn
from utils import *
import copy


def run_I2Attack(attack_model, src_data, tgt_data, device, eps=0, step_size=0, attack_epochs=0):
    src_examples = src_data['X']
    src_labels = src_data['Y']
    p_src_examples = np.clip(src_examples + np.random.uniform(-eps, eps), a_min=src_examples.min(), a_max=src_examples.max())

    batch_size = 512 # 256 for DAN, 256 for others with SVHN -> MNIST
    attacker = I2Attack(eps=eps, step_size=step_size, device=device, min_value=src_examples.min(), max_value=src_examples.max())

    for epoch in range(attack_epochs):
        print(epoch)
        p_s_data = []
        num_blocks = int(src_examples.shape[0] / batch_size) + 1
        for k in range(num_blocks):
            if (k + 1) * batch_size < src_examples.shape[0]:
                sinputs = src_examples[k * batch_size: (k + 1) * batch_size, :]
                slabels = src_labels[k * batch_size: (k + 1) * batch_size]
                p_sinputs = p_src_examples[k * batch_size: (k + 1) * batch_size, :]
            else:
                sinputs = src_examples[k * batch_size:, :]
                slabels = src_labels[k * batch_size:]
                p_sinputs = p_src_examples[k * batch_size:, :]

            ridx = np.random.choice(tgt_data['X'].shape[0], sinputs.shape[0])
            tinputs = tgt_data['X'][ridx, :]

            sinputs = torch.tensor(sinputs, requires_grad=False, dtype=torch.float).to(device)
            slabels = torch.tensor(slabels, requires_grad=False, dtype=torch.long).to(device)
            tinputs = torch.tensor(tinputs, requires_grad=False, dtype=torch.float).to(device)
            p_sinputs = torch.tensor(p_sinputs, requires_grad=False, dtype=torch.float).to(device)
            p_sinputs = attacker(attack_model, sinputs, slabels, tinputs, p_sinputs)
            p_s_data.append(p_sinputs.detach())
        p_src_examples = torch.cat(p_s_data, dim=0)

    p_src_data = {}
    p_src_data['X'] = p_src_examples
    p_src_data['Y'] = src_data['Y']

    return p_src_data


class I2Attack(nn.Module):
    def __init__(self, eps=0., step_size=0., device=None, min_value=0, max_value=1, inner_epoch=1, update_lr=0.001):
        super(I2Attack, self).__init__()
        self.inner_epoch = inner_epoch
        self.update_lr = update_lr
        self.device = device
        self.model = None
        self.optimizer = None
        self.eps = eps
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, raw_model, sinputs, soutputs, tinputs, p_sinputs):
        self.model = raw_model
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.update_lr)
        self.model.train()
        data_grad = self.grad_estimate(sinputs, soutputs, tinputs, p_sinputs)
        eta = torch.clamp(p_sinputs + self.step_size * data_grad.sign() - sinputs, min=-self.eps, max=self.eps).detach()
        output = torch.clamp(sinputs + eta, min=self.min_value, max=self.max_value)
        return output

    def grad_estimate(self, sinputs, soutputs, tinputs, p_sinputs):
        meta_grad_sum = torch.zeros_like(p_sinputs, device=self.device, requires_grad=False)
        for iter in range(self.inner_epoch):
            p_sinputs.requires_grad = False
            self.optimizer.zero_grad()
            loss = self.model(p_sinputs, soutputs, tinputs)
            loss.backward()
            self.optimizer.step()

            self.model.zero_grad()
            p_sinputs.requires_grad = True
            meta_loss = self.model.attack_obj(sinputs, soutputs, p_sinputs)
            meta_grad_sum += torch.autograd.grad(meta_loss, p_sinputs, retain_graph=False, create_graph=False)[0]
        return meta_grad_sum
