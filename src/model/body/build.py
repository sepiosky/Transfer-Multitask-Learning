from yacs.config import CfgNode
from torch import nn
from src.utils.registry import Registry

BODY_REGISTRY = Registry()

def build_body(body_cfg: CfgNode) -> nn.Module:
    body_module = BODY_REGISTRY[body_cfg.NAME](body_cfg)
    return body_module