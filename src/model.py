import timm, torch.nn as nn, torchvision
from config import BACKBONE, DROPOUT_P

def build_model(backbone=BACKBONE, dropout=DROPOUT_P):
    if backbone.startswith("x3d"):
        mdl = torchvision.models.video.__dict__[backbone](weights="KINETICS400_V1")
        in_f = mdl.fc.in_features; mdl.fc = nn.Identity()
    else:
        mdl = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        in_f = mdl.num_features
    head = nn.Sequential(nn.Dropout(dropout), nn.BatchNorm1d(in_f), nn.Linear(in_f,1))
    return nn.Sequential(mdl, head)
