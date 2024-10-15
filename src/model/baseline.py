import torch
from torch import nn
import numpy as np
from yacs.config import CfgNode
from src.model.backbone.build import build_backbone
from src.model.body.build import build_body
from src.model.heads.build import build_heads

from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register('baseline')
def build_baseline_model(model_cfg: CfgNode) -> nn.Module:
    """
    Builds the baseline model using the CFGNode object,
    :param model_cfg: YAML based YACS configuration node.
    :return: return torch neural network module
    """
    # instantiate and return the BaselineModel using the configuration node
    return BaselineModel(model_cfg)


# Baseline model class must extends torch.nn.module
class BaselineModel(nn.Module):

    def __init__(self, model_cfg: CfgNode):
        """
        # Constructor will take a CFG node to read key properties from it.
        :param model_cfg:
        """
        # Cell the super constructor
        super(BaselineModel, self).__init__()

        # Build backbone using backbone dict of property
        self.backbone = build_backbone(model_cfg.BACKBONE)

        # Build body using body dict of properties
        self.body = build_body(model_cfg.BODY)

        # Build heads using heads dict of properties
        self.heads = build_heads(model_cfg.HEADS)

        # Store output dimensions in the base model instance variables
        #self.body_output_dims = model_cfg.BODY.OUTPUT_DIMS

    def forward(self, x):
        """
        Build backbone, build head, return it.
        :param x:
        :return:
        """
        # Call build_backbone function on input
        x = self.backbone(x)

        # Call build_body function on the output above
        x = self.body(x)

        # split output into the #num_heads classes
        #logits = list(torch.split(x, self.body_output_dims, dim=1))
        logits = [None for _ in range(len(self.heads))] if len(self.heads) > 0 else [x]
        # Call build_heads function on the output above
        for i in range(len(self.heads)):
            #logits[i] = self.heads[i](logits[i])
            logits[i] = self.heads[i](x)

        return logits

    def freeze_bn(self):
        """
        Freeze BatchNorm layers
        :return:
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()