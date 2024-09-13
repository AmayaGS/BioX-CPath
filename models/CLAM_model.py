
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):

    """
    L: input feature dimension
    D: hidden layer dimension
    Dropout: True or False
    n_classes: number of classes
    """

    def __init__(self, L= 1024, D=224, Dropout=True, dropout_rate = 0.25, n_classes=2, k_sample=8, instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):

        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.Dropout= Dropout
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.k_sample = k_sample

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        if self.Dropout:
            self.attention_V.append(nn.Dropout(self.dropout_rate))
            self.attention_U.append(nn.Dropout(self.dropout_rate))

        self.attention_V = nn.Sequential(*self.attention_V)
        self.attention_U = nn.Sequential(*self.attention_U)

        self.attention_weights = nn.Linear(self.D, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.n_classes)
        )

        instance_classifiers = [nn.Linear(self.L, 2) for i in range(n_classes)] #  n_classes?
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

    def forward(self, x, label, instance_eval=True):

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD

        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK ##################
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, x, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, x, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, x)  # KxL

        logits = self.classifier(M)
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'logits': logits, 'Y_prob': Y_prob, 'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {'logits': logits, 'Y_prob': Y_prob}

        return logits, Y_prob, results_dict, label

    # AUXILIARY METHODS
    def calculate_error(self, Y_hat, Y):
        error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
        return error

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, x, classifier):  # h=x
        device=x.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(x, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, x, classifier):
        device=x.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets