import torch
from yacs.config import CfgNode
from src.model.backbone.build import build_backbone

def get_backbone_embeddings_size(backbone_cfg: CfgNode, input_size = (1,28,28)):
    model = build_backbone(backbone_cfg)
    device = next(model.parameters()).device
    x = torch.rand([2,*input_size]).to(device)
    with torch.no_grad():
        output = model.forward(x)
        emb_size = output.shape[-1]

    del x,output,model
    torch.cuda.empty_cache()

    return emb_size