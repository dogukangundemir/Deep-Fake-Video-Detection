import os, json, random, numpy as np, torch, argparse, wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset import DFDCFrames
from model   import build_model
from config  import *

def mixup(x,y,alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha,alpha); idx = torch.randperm(x.size(0))
    return lam*x+(1-lam)*x[idx], lam*y+(1-lam)*y[idx]

class SmoothBCELoss(torch.nn.Module):
    def __init__(self, eps=LABEL_SMOOTH):
        super().__init__(); self.eps=eps
        self.bce = torch.nn.BCEWithLogitsLoss()
    def forward(self, logit, target):
        target = target*(1-self.eps)+0.5*self.eps; return self.bce(logit,target)

def run(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = DFDCFrames(a.train_csv, a.frames_dir+"/train", "train")
    val_ds   = DFDCFrames(a.val_csv,   a.frames_dir+"/train", "val")
    tr_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    vl_dl = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,num_workers=4)

    net = build_model(a.backbone, a.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, EPOCHS)
    crit = SmoothBCELoss(); scaler = GradScaler(enabled=AMP)

    wandb.init(project="dfdc-detector", config=vars(a))
    writer = SummaryWriter("outputs/tb")

    best_auc = 0
    for epoch in range(EPOCHS):
        net.train(); total=0; running=0
        for x,y in tr_dl:
            x,y = x.to(dev), y.to(dev); x,y = mixup(x,y)
            with autocast(enabled=AMP):
                loss = crit(net(x).squeeze(1), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            total+=x.size(0); running+=loss.item()*x.size(0)
        sched.step()

        net.eval(); preds,targets = [],[]
        with torch.no_grad():
            for x,y in vl_dl:
                p = torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist()
                preds+=p; targets+=y.tolist()
        auc = roc_auc_score(targets,preds)
        writer.add_scalar("val/auc", auc, epoch); wandb.log({"val_auc":auc})
        if auc>best_auc:
            best_auc=auc; os.makedirs("outputs/checkpoints",exist_ok=True)
            torch.save(net.state_dict(), "outputs/checkpoints/best.pt")
        print(f"Epoch {epoch} AUC {auc:.3f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--train-csv', required=True)
    p.add_argument('--val-csv',   required=True)
    p.add_argument('--frames-dir',required=True, default='data/frames')
    p.add_argument('--backbone',  default=BACKBONE)
    p.add_argument('--dropout',   type=float, default=DROPOUT_P)
    run(p.parse_args())
