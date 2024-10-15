import torch.nn as nn
import torch

from .build import BACKBONE_REGISTRY

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNetBack(nn.Module):
    def __init__(self, input_channels = 1):
        super(ProtoNetBack, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1),
            conv_block(64, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            conv_block(128, 128),
        )

    def get_embedding_size(self, input_size = (1,28,28)):
        device = next(self.parameters()).device
        x = torch.rand([2,*input_size]).to(device)
        with torch.no_grad():
            output = self.forward(x)
            emb_size = output.shape[-1]

        del x,output
        torch.cuda.empty_cache()

        return emb_size

    def forward(self, x):
        return self.layers(x).reshape([x.shape[0] , -1])


@BACKBONE_REGISTRY.register('protonet')
def build_protonet_backbone(backbone_cfg):
    """
    :param backbone_cfg: backbone config node
    :param kwargs:
    :return: backbone module
    """
    model = ProtoNetBack(input_channels = backbone_cfg.INPUT_CHANNELS)
    if backbone_cfg.get('PRETRAINED_PATH') != '':
        pretrained_path = backbone_cfg['PRETRAINED_PATH']
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model