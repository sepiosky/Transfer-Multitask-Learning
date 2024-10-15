from torchvision import models
from src.utils.registry import Registry
from torch import nn
from yacs.config import CfgNode

BACKBONE_REGISTRY = Registry()

def build_backbone(backbone_cfg: CfgNode) -> nn.Module:
    """
    build backbone
    :param backbone_cfg:
    :return:
    """
    backbone = BACKBONE_REGISTRY[backbone_cfg.NAME](backbone_cfg)

    if backbone_cfg.FREEZE:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone
