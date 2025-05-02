import os, random, json, argparse, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from config import *
from dataset import DFDCFrames
from model   import build_model

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

class SmoothBCE(torch.nn.Module):
    def __init__(self, eps=LBL_SMOOTH): super().__init__(); self.eps=eps
    def forward(self, logits, target):
        target = target*(1-self.eps)+0.5*self.eps
        return torch.nn.functional.binary_cross_entropy_with_logits(logits,target)

def mixup(x,y,a=MIXUP_A):
    lam = np.random.beta(a,a); idx=torch.randperm(x.size(0))
    return lam*x+(1-lam)*x[idx], lam*y+(1-lam)*y[idx]

def main(a):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_ds=DFDCFrames(a.train_csv,a.image_root,"train")
    vl_ds=DFDCFrames(a.val_csv,  a.image_root,"val")
    tr_dl=DataLoader(tr_ds,BATCH_SIZE,shuffle=True,num_workers=4)
    vl_dl=DataLoader(vl_ds,BATCH_SIZE,shuffle=False,num_workers=4)

    net=build_model(a.backbone,a.dropout).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=LR)
    sched=CosineAnnealingLR(opt,EPOCHS)
    crit=SmoothBCE(); scaler=GradScaler(enabled=AMP)
    writer=SummaryWriter("outputs/tb"); best=0

    history={'epoch':[],'train_loss':[],'val_auc':[]}

    for ep in range(EPOCHS):
        net.train(); running=0; total=0
        for x,y in tr_dl:
            x,y = x.to(dev),y.to(dev); x,y = mixup(x,y)
            with autocast(enabled=AMP):
                loss=crit(net(x).squeeze(1),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            running+=loss.item()*x.size(0); total+=x.size(0)
        sched.step()

        net.eval(); preds,truth=[],[]
        with torch.no_grad():
            for x,y in vl_dl:
                p=torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist()
                preds+=p; truth+=y.tolist()
        auc=roc_auc_score(truth,preds); print(f"Epoch {ep} AUC {auc:.3f}")
        history['epoch'].append(ep)
        history['train_loss'].append(running/total)
        history['val_auc'].append(auc)
        writer.add_scalar("val/auc",auc,ep)

        if auc>best:
            best=auc; os.makedirs("outputs/checkpoints",exist_ok=True)
            torch.save(net.state_dict(),"outputs/checkpoints/best.pt")

    json.dump(history, open("outputs/history.json","w"), indent=2)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv',   required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--backbone',  default=BACKBONE)
    ap.add_argument('--dropout',   type=float, default=DROPOUT)
    main(ap.parse_args())
