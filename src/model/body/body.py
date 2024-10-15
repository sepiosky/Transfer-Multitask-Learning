from torch import nn
from yacs.config import CfgNode
from src.model.custom_layers.linear import CustomLinearLayer
from .build import BODY_REGISTRY

@BODY_REGISTRY.register('simple_body')
def build_simple_body(body_cfg: CfgNode) -> nn.Module:
    return SimpleBody(body_cfg)


class SimpleBody(nn.Module):

    def __init__(self, body_cfg: CfgNode):
        super(SimpleBody, self).__init__()
        self.fc_layers = []
        input_dim = body_cfg.INPUT_DIM
        # first hidden layers
        for idx, hidden_dim in enumerate(body_cfg.HIDDEN_DIMS):
            layer = CustomLinearLayer(input_dim, hidden_dim, bn=body_cfg.BN, activation=body_cfg.ACTIVATION,
                            dropout_rate=body_cfg.DROPOUT)
            if body_cfg.HIDDEN_DIMS_FREEZE[idx]:
                for param in layer.parameters():
                    param.requires_grad = False
            self.fc_layers.append(layer)
            input_dim = hidden_dim

        output_dim = body_cfg.OUTPUT_DIM

        # prediction layer
        output_layer = CustomLinearLayer(input_dim, output_dim, bn=False, activation=None, dropout_rate=-1)
        if body_cfg.HIDDEN_DIMS_FREEZE[-1]:
            for param in output_layer.parameters():
                param.requires_grad = False
        self.fc_layers.append(output_layer)

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