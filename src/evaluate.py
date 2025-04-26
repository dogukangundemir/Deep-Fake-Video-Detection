import argparse, os, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader
from dataset import DFDCFrames
from model import build_model
from config import BACKBONE, DROPOUT_P

def main(a):
    ds = DFDCFrames(a.csv, a.frames_dir+"/train", "val")
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    dev=torch.device(a.device)
    net=build_model(a.backbone, a.dropout); net.load_state_dict(torch.load(a.model, map_location=dev)); net.to(dev).eval()
    preds, y = [], []
    with torch.no_grad():
        for x,t in dl:
            preds+=torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist(); y+=t.tolist()

    os.makedirs(a.out, exist_ok=True)
    # ROC
    fpr,tpr,_=roc_curve(y,preds); roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC AUC={roc_auc:.3f}")
    plt.tight_layout(); plt.savefig(f"{a.out}/roc.png"); plt.close()
    # CM
    cm=confusion_matrix(y,[p>0.5 for p in preds]); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.tight_layout(); plt.savefig(f"{a.out}/cm.png"); plt.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--csv',   required=True)
    p.add_argument('--frames-dir',default='data/frames')
    p.add_argument('--out',   default='outputs/full_eval')
    p.add_argument('--device',default='cuda')
    p.add_argument('--backbone',default=BACKBONE)
    p.add_argument('--dropout', type=float, default=DROPOUT_P)
    main(p.parse_args())
