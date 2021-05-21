import torch
import torch.nn as nn
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
    def __init__(self, num_classes):
        super(DANNModel, self).__init__()
        self.num_classes = num_classes
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_extractor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier_layer = nn.Linear(512, num_classes)

        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s_inputs, s_outputs, t_inputs, alpha=1.0):
        s_feats = self.fc_extractor(self.conv_extractor(s_inputs).view(s_inputs.shape[0], -1))
        t_feats = self.fc_extractor(self.conv_extractor(t_inputs).view(t_inputs.shape[0], -1))
        s_preds = self.classifier_layer(s_feats)
        class_loss = self.criterion(s_preds, s_outputs)

        domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_feats, t_feats], dim=0), alpha))
        domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
        domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)
        domain_loss = self.criterion(domain_preds, domain_labels)
        loss = class_loss + domain_loss
        return loss

    def inference(self, x):
        x = self.fc_extractor(self.conv_extractor(x).view(x.shape[0], -1))
        return self.classifier_layer(x)

    def get_discrepancy(self, s_inputs, t_inputs):
        s_feats = self.fc_extractor(self.conv_extractor(s_inputs).view(s_inputs.shape[0], -1))
        t_feats = self.fc_extractor(self.conv_extractor(t_inputs).view(t_inputs.shape[0], -1))
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
        sfeats = self.fc_extractor(self.conv_extractor(sinputs).view(sinputs.shape[0], -1))
        p_sfeats = self.fc_extractor(self.conv_extractor(p_sinputs).view(p_sinputs.shape[0], -1))
        Ys_onehot = self.one_hot_embedding(soutputs, num_classes=self.num_classes)
        ratio = 10000
        lab_attack_loss = mmd.mmd_rbf_noaccelerate(torch.cat([sfeats, ratio * Ys_onehot], dim=1),
                                                   torch.cat([p_sfeats, ratio * Ys_onehot], dim=1))

        marg_loss = mmd.mmd_rbf_noaccelerate(sfeats, p_sfeats)
        all_s_preds = self.classifier_layer(p_sfeats)
        class_loss = self.criterion(all_s_preds, soutputs)

        return lab_attack_loss - (marg_loss*10 + class_loss) * 2e-6
