from yacs.config import CfgNode
from torch import nn
from src.utils.registry import Registry

HEADS_REGISTRY = Registry()

def build_heads(heads_cfg: CfgNode) -> nn.Module:
    heads_module_list = nn.ModuleList([ HEADS_REGISTRY[heads_cfg.NAMES[idx]](heads_cfg, idx) for idx in range(len(heads_cfg.INPUT_DIMS)) ])
    return heads_module_list