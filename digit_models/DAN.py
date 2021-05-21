import torch
import torch.nn as nn
from models import mmd
import numpy as np


class DANModel(nn.Module):
    def __init__(self, num_classes):
        super(DANModel, self).__init__()
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

        mmd_loss = mmd.mmd_rbf_noaccelerate(s_feats, t_feats)
        loss = class_loss + alpha * mmd_loss
        return loss

    def get_discrepancy(self, s_inputs, t_inputs):
        s_feats = self.fc_extractor(self.conv_extractor(s_inputs).view(s_inputs.shape[0], -1))
        t_feats = self.fc_extractor(self.conv_extractor(t_inputs).view(t_inputs.shape[0], -1))
        mmd_loss = mmd.mmd_rbf_noaccelerate(s_feats, t_feats)
        return mmd_loss

    def inference(self, x):
        x = self.fc_extractor(self.conv_extractor(x).view(x.shape[0], -1))
        return self.classifier_layer(x)

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

        all_sfeats = self.fc_extractor(self.conv_extractor(sinputs).view(sinputs.shape[0], -1))
        all_p_sfeats = self.fc_extractor(self.conv_extractor(p_sinputs).view(p_sinputs.shape[0], -1))
        marg_loss = mmd.mmd_rbf_noaccelerate(all_sfeats, all_p_sfeats)
        all_s_preds = self.classifier_layer(all_p_sfeats)
        class_loss = self.criterion(all_s_preds, soutputs)

        return lab_attack_loss - (marg_loss * 10 + class_loss) * 2e-6
        # return lab_attack_loss - (marg_loss + class_loss) * 2e-6
