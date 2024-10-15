import torch
from torch import nn
from yacs.config import CfgNode
from src.utils.registry import Registry
import os
#from src.models.layers.sync_batchnorm import convert_model

META_ARCH_REGISTRY = Registry()


def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    build model
    :param model_cfg: model config blob
    :return: model
    """
    meta_arch = model_cfg.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(model_cfg)

    if model_cfg.get('PRETRAINED_PATH') != '':
        pretrained_path = os.path.abspath(model_cfg['PRETRAINED_PATH'])
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state'], strict=False)

    # This is VERY SLOW
    # if model_cfg.NORMALIZATION_FN == 'SYNC_BN':
    #     model = convert_model(model)

    return model