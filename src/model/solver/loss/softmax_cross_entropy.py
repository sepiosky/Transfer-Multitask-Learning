from audioop import mul
import torch
from typing import List, Dict, Union
from yacs.config import CfgNode
from .build import LOSS_REGISTRY
import numpy as np


@LOSS_REGISTRY.register('xentropy')
class SoftmaxCE(torch.nn.Module):
    """
    Normal softmax cross entropy
    """

    def __init__(self, loss_cfg: CfgNode, **kwargs):
        """
        :param weights: class weights (deleted)
        """
        super(SoftmaxCE, self).__init__()

        self.ohem_rate = loss_cfg.OHEM_RATE
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        multihead_labels = labels.detach().clone().cpu()
        multihead_labels[ multihead_labels == -1 ] = 0

        loss = 0

        preds = torch.argmax(logits.float(), dim=1)
        losses = self.loss_fn(logits, multihead_labels) # self.loss_fn(logits, torch.tensor(multihead_labels))
        head_mask = [0 if l==-1 else 1 for l in labels]
        losses = torch.mul(losses, torch.tensor(head_mask, requires_grad=False, device=losses.device))
        corrects = (labels == preds) # so -1 dont count
        if self.ohem_rate < 1:
            loss += self.compute_ohem_loss(losses)
        else:
            loss += losses.mean()

        num_head_samples = len(labels[labels!=-1])
        acc = -1 if num_head_samples==0 else torch.sum(corrects) / (num_head_samples + 0.0)

        return loss, acc, num_head_samples

    def compute_ohem_loss(self, losses: torch.Tensor):
        N = losses.shape[0]
        keep_size = int(N * self.ohem_rate)
        ohem_losses, _ = losses.topk(keep_size)
        loss = ohem_losses.mean()
        return loss