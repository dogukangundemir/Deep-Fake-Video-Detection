import torch.nn as nn, timm, torchvision
from config import BACKBONE, DROPOUT

def build_model(backbone=BACKBONE, dropout=DROPOUT):
    if backbone.startswith("x3d"):
        base = torchvision.models.video.__dict__[backbone](weights="KINETICS400_V1")
        in_f = base.fc.in_features; base.fc = nn.Identity()
    else:
        base = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        in_f = base.num_features
    head = nn.Sequential(nn.Dropout(dropout), nn.BatchNorm1d(in_f), nn.Linear(in_f,1))
    return nn.Sequential(base, head)
