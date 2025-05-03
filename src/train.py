import os, random, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from config import (
    BACKBONE, DROPOUT, LR, EPOCHS, BATCH_SIZE,
    MIXUP_A, LBL_SMOOTH, AMP, SEED,
    FREEZE_EPOCHS, UNFROZEN_LR            
)
from dataset import DFDCFrames
from model import build_model

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def set_trainable(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

class SmoothBCE(torch.nn.Module):
    def __init__(self, eps=LBL_SMOOTH):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        target = target*(1-self.eps) + 0.5*self.eps
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

def mixup(x, y, alpha=MIXUP_A):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], lam*y + (1-lam)*y[idx]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    train_ds = DFDCFrames(args.train_csv, args.image_root, mode="train")
    val_ds   = DFDCFrames(args.val_csv,   args.image_root, mode="val")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)

    
    net = build_model(args.backbone, args.dropout).to(device)

    set_trainable(net.base, False)
    optimizer = torch.optim.AdamW(net.head.parameters(), lr=LR)

    backbone_added = False
    scheduler = CosineAnnealingLR(optimizer, EPOCHS)

    criterion = SmoothBCE()
    scaler = GradScaler(enabled=AMP)
    
    writer = SummaryWriter("outputs/tb")
    best_auc = 0
    history = {'epoch':[], 'train_loss':[], 'val_auc':[]}

    for epoch in range(EPOCHS):
        
        net.train()
        running_loss, total_samples = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = mixup(x, y)
            with autocast(enabled=AMP):
                logits = net(x).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        scheduler.step()
        train_loss = running_loss / total_samples

        
        net.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                p = torch.sigmoid(net(x).squeeze(1)).cpu().tolist()
                val_preds += p
                val_trues += y.tolist()

        val_auc = roc_auc_score(val_trues, val_preds)
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_auc={val_auc:.4f}")

        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/auc",    val_auc,     epoch)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)

        
        if (epoch + 1) == FREEZE_EPOCHS and not backbone_added:
            print(f"🔓 Unfreezing backbone at epoch {epoch+1}")
            set_trainable(net.base, True)
            optimizer.add_param_group({
                "params": net.base.parameters(),
                "lr": UNFROZEN_LR
            })
            backbone_added = True

       
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs("outputs/checkpoints", exist_ok=True)
            torch.save(net.state_dict(), "outputs/checkpoints/best.pt")


    with open("outputs/history.json","w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv',  required=True)
    p.add_argument('--val-csv',    required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--backbone',   default=BACKBONE)
    p.add_argument('--dropout',    type=float, default=DROPOUT)
    args = p.parse_args()
    main(args)