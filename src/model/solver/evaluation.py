from operator import mod
import torch
import pickle
import numpy as np
from torch import nn
from collections import Counter
from yacs.config import CfgNode
from typing import List, Dict, Union
from sklearn.metrics import classification_report
#from .loss import WeightedFocalLoss, SoftMaxCE
from .loss import build_loss

class MultiHeadsEvaluation(nn.Module):
    def __init__(self, model_cfg: CfgNode):
        super(MultiHeadsEvaluation, self).__init__()
        solver_cfg = model_cfg.SOLVER
        loss_cfg = solver_cfg.LOSS
        heads_cfg = model_cfg.HEADS

        self.loss_fns = [build_loss(loss_cfg, eps=loss_cfg.EPS, reduction=loss_cfg.REDUCTION, num_classes=heads_cfg.OUTPUT_DIMS[head_idx]) for head_idx in range(len(heads_cfg.OUTPUT_DIMS)) ]
        # self.num_heads = len(heads_cfg.OUTPUT_DIMS) #works wrong when heads are not connected
        self.num_heads = None

        self.logits_caches = None
        self.labels_cache = None
        self.acc_cache = []
        self.loss_cache = []

    def set_num_heads(self, num_heads):
        ## Alert: if we dont use this if in each batch logits cashes will be cleared in train script
        if self.num_heads is None:
            self.logits_caches = [ [] for _ in range(num_heads) ]
            self.labels_caches = [ [] for _ in range(num_heads) ]
        self.num_heads = num_heads

    def forward(self, heads_logits: list, labels: Union[tuple, torch.Tensor]) -> Dict:

        if self.num_heads > 1:
            heads_labels = [labels[:, head_idx] for head_idx in range(self.num_heads) ]
        else:
            heads_labels = [labels]

        heads_losses, heads_accs, num_samples = [], [], [] # num_samples is not size of batch. in multi label or multi task may differ (some heads get -1 labels so its not that heads sample)
        for head_idx in range(self.num_heads):
            loss, acc, num_head_samples = self.loss_fns[head_idx](heads_logits[head_idx], heads_labels[head_idx])
            heads_losses.append(loss)
            heads_accs.append(acc)
            num_samples.append(num_head_samples)

        loss = sum(heads_losses)
        acc = sum([a*b for a,b in zip(heads_accs,num_samples)]) / sum(num_samples)

        eval_result = {
            'heads_losses': heads_losses,
            'heads_accs': heads_accs,
            'loss': loss,
            'acc': acc
        }
        # dump data in cache
        for head_idx in range(self.num_heads):
            self.logits_caches[head_idx].append(heads_logits[head_idx].detach().cpu().numpy())
            self.labels_caches[head_idx].append(heads_labels[head_idx].detach().cpu().numpy()) #TODO checkk
        self.loss_cache.append(loss.detach().item())
        self.acc_cache.append(acc.detach().item())
        return eval_result

    def clear_cache(self):
        self.logits_caches = [ [] for _ in range(self.num_heads) ]
        self.labels_caches = [ [] for _ in range(self.num_heads) ]
        self.loss_cache = []
        self.acc_cache = []

    def evalulate_on_cache(self):
        heads_logits_all = [ np.vstack(self.logits_caches[head_idx]) for head_idx in range(self.num_heads) ]
        labels_all = [ np.hstack(self.labels_caches[head_idx]) for head_idx in range(self.num_heads) ]

        heads_preds = [ np.argmax(heads_logits_all[head_idx], axis=1) for head_idx in range(self.num_heads) ]

        heads_clf_result = [ classification_report(labels_all[head_idx], heads_preds[head_idx], output_dict=True, zero_division=0) for head_idx in range(self.num_heads) ]

        preds_labels = []
        for idx, _ in enumerate(heads_preds[0]):
            labels = [ labels_all[head_idx][idx] for head_idx in range(self.num_heads) ]
            entry = {
                'preds': [heads_preds[head_idx][idx] for head_idx in range(self.num_heads) ],
                'labels': labels
            }
            preds_labels.append(entry)

        acc = np.mean(self.acc_cache)
        loss = np.mean(self.loss_cache)
        result = {
            'clf_results': heads_clf_result,
            'preds_labels': preds_labels,
            'acc': acc,
            'loss': loss
        }
        return result


def build_evaluator(solver_cfg: CfgNode):
    return MultiHeadsEvaluation(solver_cfg)