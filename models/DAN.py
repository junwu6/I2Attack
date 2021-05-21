import torch.nn as nn
from models import mmd
from network.resnet import resnet50
import torch


class DANModel(nn.Module):
    def __init__(self, num_classes, use_bottleneck=True, bottleneck_width=256, width=1024):
        super(DANModel, self).__init__()
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

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s_inputs, s_outputs, t_inputs, alpha=1.0):
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
            t_feats = self.bottleneck_layer(t_feats)
        s_preds = self.classifier_layer(s_feats)
        class_loss = self.criterion(s_preds, s_outputs)
        mmd_loss = mmd.mmd_rbf_noaccelerate(s_feats, t_feats)

        loss = class_loss + alpha * mmd_loss
        return loss

    def get_discrepancy(self, s_inputs, t_inputs):
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
            t_feats = self.bottleneck_layer(t_feats)
        mmd_loss = mmd.mmd_rbf_noaccelerate(s_feats, t_feats)
        return mmd_loss

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)

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
        return lab_attack_loss - (marg_loss + class_loss) * 2



        # sfeats = self.base_network(sinputs).view(sinputs.shape[0], -1)
        # p_sfeats = self.base_network(p_sinputs).view(p_sinputs.shape[0], -1)
        # if self.use_bottleneck:
        #     sfeats = self.bottleneck_layer(sfeats)
        #     p_sfeats = self.bottleneck_layer(p_sfeats)
        # Ys_onehot = self.one_hot_embedding(soutputs, num_classes=self.num_classes)
        # ratio = 10000
        # lab_attack_loss = mmd.mmd_rbf_noaccelerate(torch.cat([sfeats, ratio * Ys_onehot], dim=1),
        #                                            torch.cat([p_sfeats, ratio * Ys_onehot], dim=1))
        #
        # # lab_attack_loss = 0.
        # # for i in range(self.num_classes):
        # #     if sum(soutputs == i) == 0: continue
        # #     sfeats = self.bottleneck_layer(self.base_network(sinputs[soutputs == i]).view(sinputs[soutputs == i].shape[0], -1))
        # #     p_sfeats = self.bottleneck_layer(self.base_network(p_sinputs[soutputs == i]).view(p_sinputs[soutputs == i].shape[0], -1))
        # #     lab_attack_loss += mmd.mmd_rbf_noaccelerate(sfeats, p_sfeats)
        #
        # marg_loss = mmd.mmd_rbf_noaccelerate(sfeats, p_sfeats)
        # all_s_preds = self.classifier_layer(p_sfeats)
        # class_loss = self.criterion(all_s_preds, soutputs)
        # # return lab_attack_loss - class_loss * 1e-6 - marg_loss * 1e-6  # for visda
        # # return lab_attack_loss - (marg_loss + class_loss) * 5e-8
        return lab_attack_loss - (marg_loss + class_loss) * 3e-8
