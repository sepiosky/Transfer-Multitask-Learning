from torch import nn
from yacs.config import CfgNode
from src.model.custom_layers.linear import CustomLinearLayer
from .build import HEADS_REGISTRY

@HEADS_REGISTRY.register('simple_head')
def build_simple_heads(heads_cfg: CfgNode, head_idx: int) -> nn.Module:
    return SimpleHead(heads_cfg, head_idx)


class SimpleHead(nn.Module):

    def __init__(self, heads_cfg: CfgNode, head_idx: int):
        super(SimpleHead, self).__init__()
        self.fc_layers = []
        input_dim = heads_cfg.INPUT_DIMS[head_idx]
        # first hidden layers
        for idx, hidden_dim in enumerate(heads_cfg.HIDDEN_DIMS[head_idx]):
            layer = CustomLinearLayer(input_dim, hidden_dim, bn=heads_cfg.BN, activation=heads_cfg.ACTIVATION,
                            dropout_rate=heads_cfg.DROPOUT)
            if heads_cfg.HIDDEN_DIMS_FREEZE[head_idx][idx]:
                for param in layer.parameters():
                    param.requires_grad = False
            self.fc_layers.append(layer)
            input_dim = hidden_dim


        # prediction layer
        output_dim = heads_cfg.OUTPUT_DIMS[head_idx]
        self.fc_layers.append(
            CustomLinearLayer(input_dim, output_dim, bn=False, activation=None, dropout_rate=-1)
        )

        self.fc_layers = nn.Sequential(*self.fc_layers)

        # TODO load pretrain params
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        return self.fc_layers(x)