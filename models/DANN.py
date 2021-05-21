import torch
import torch.nn as nn
from network.resnet import resnet50
from torch.autograd import Function
import numpy as np
from models import mmd


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNModel(nn.Module):
    def __init__(self, num_classes, use_bottleneck=True, bottleneck_width=256, width=1024):
        super(DANNModel, self).__init__()
        self.base_network = resnet50(pretrained=True)
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(2048, bottleneck_width),
            nn.BatchNorm1d(bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        hidden_dim = bottleneck_width if self.use_bottleneck else 2048
        self.classifier_layer = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 2),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s_inputs, s_outputs, t_inputs, alpha=1.0):
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
            t_feats = self.bottleneck_layer(t_feats)
        s_preds = self.classifier_layer(s_feats)
        class_loss = self.criterion(s_preds, s_outputs)
        domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_feats, t_feats], dim=0), alpha))
        domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
        domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)
        domain_loss = self.criterion(domain_preds, domain_labels)
        loss = class_loss + domain_loss
        return loss

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)

    def get_discrepancy(self, s_inputs, t_inputs):
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
            t_feats = self.bottleneck_layer(t_feats)
        outputs = self.discriminator(ReverseLayerF.apply(torch.cat([s_feats, t_feats], dim=0), 1.0))
        domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
        domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)

        preds = torch.max(outputs, 1)[1]
        domain_acc = torch.sum(preds == domain_labels)
        domain_acc = domain_acc.double() / outputs.shape[0]
        return max(domain_acc, 1-domain_acc)

    def one_hot_embedding(self, labels, num_classes):
        y = torch.eye(num_classes, device=labels.device)
        return y[labels]

    def attack_obj(self, sinputs, soutputs, p_sinputs):
        all_sfeats = self.bottleneck_layer(self.base_network(sinputs).view(sinputs.shape[0], -1))
        all_p_sfeats = self.bottleneck_layer(self.base_network(p_sinputs).view(p_sinputs.shape[0], -1))
        lab_attack_loss = 0.
        for i in range(self.num_classes):
            if sum(soutputs == i) == 0: continue
            sfeats = all_sfeats[soutputs == i]
            p_sfeats = all_p_sfeats[soutputs == i]
            lab_attack_loss += mmd.mmd_rbf_noaccelerate(sfeats, p_sfeats)
        marg_loss = mmd.mmd_rbf_noaccelerate(all_sfeats, all_p_sfeats)
        all_s_preds = self.classifier_layer(all_p_sfeats)
        class_loss = self.criterion(all_s_preds, soutputs)
        return lab_attack_loss - (marg_loss + class_loss) * 1

        # sfeats = self.base_network(sinputs).view(sinputs.shape[0], -1)
        # p_sfeats = self.base_network(p_sinputs).view(p_sinputs.shape[0], -1)
        # if self.use_bottleneck:
        #     sfeats = self.bottleneck_layer(sfeats)
        #     p_sfeats = self.bottleneck_layer(p_sfeats)
        # Ys_onehot = self.one_hot_embedding(soutputs, num_classes=self.num_classes)
        # ratio = 10000
        # lab_attack_loss = mmd.mmd_rbf_noaccelerate(torch.cat([sfeats, ratio * Ys_onehot], dim=1), torch.cat([p_sfeats, ratio * Ys_onehot], dim=1))
        #
        # marg_loss = mmd.mmd_rbf_noaccelerate(sfeats, p_sfeats)
        # all_s_preds = self.classifier_layer(p_sfeats)
        # class_loss = self.criterion(all_s_preds, soutputs)
        # return lab_attack_loss - class_loss * 5e-7  # for visda
        # # return lab_attack_loss - (marg_loss + class_loss) * 3e-8
