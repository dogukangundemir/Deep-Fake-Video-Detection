import os, argparse, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix, accuracy_score, f1_score, roc_auc_score)
from torch.utils.data import DataLoader
from dataset import DFDCFrames
from model import build_model
from config import BACKBONE, DROPOUT, BATCH_SIZE

def save_grid(df, root, tag, path, top_n=9, largest=True):
    sub=(df[df.true==(0 if tag=='FP' else 1)]
         .sort_values('pred',ascending=not largest).head(top_n))
    cols=int(np.sqrt(top_n)); rows=int(np.ceil(top_n/cols))
    plt.figure(figsize=(cols*3,rows*3))
    for i,r in enumerate(sub.itertuples()):
        img=plt.imread(os.path.join(root,r.video_id,f"{r.frame}.jpg"))
        ax=plt.subplot(rows,cols,i+1); ax.imshow(img); ax.axis('off')
        ax.set_title(f"p={r.pred:.2f}")
    plt.suptitle(tag); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def main(a):
    os.makedirs(a.out,exist_ok=True)
    dev=torch.device(a.device)
    ds = DFDCFrames(a.val_csv, a.image_root, "val")
    dl = DataLoader(ds,BATCH_SIZE,shuffle=False)
    net=build_model(a.backbone,a.dropout); net.load_state_dict(torch.load(a.model,map_location=dev))
    net.to(dev).eval()

    preds,truth=[],[]
    with torch.no_grad():
        for x,y in dl:
            preds+=torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist()
            truth+=y.tolist()
    df_val=pd.read_csv(a.val_csv); df_val['pred']=preds; df_val['true']=truth

    # roc
    fpr,tpr,_=roc_curve(truth,preds); auc=roc_auc_score(truth,preds)
    plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC AUC={auc:.3f}")
    plt.savefig(f"{a.out}/roc_curve.png",dpi=300); plt.close()\
        
    # pr
    prec,rec,_=precision_recall_curve(truth,preds); ap=average_precision_score(truth,preds)
    plt.plot(rec,prec); plt.title(f"PR AP={ap:.3f}"); plt.savefig(f"{a.out}/pr_curve.png",dpi=300); plt.close()
    
    # cm
    cm=confusion_matrix(truth,np.array(preds)>0.5)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues'); plt.title('Confusion'); plt.savefig(f"{a.out}/cmatrix.png",dpi=300); plt.close()
    
    # metrics bar
    acc=accuracy_score(truth,np.array(preds)>0.5); f1=f1_score(truth,np.array(preds)>0.5)
    sns.barplot(x=['AUC','ACC','F1'],y=[auc,acc,f1]); plt.ylim(0,1); plt.title('Score summary')
    plt.savefig(f"{a.out}/metrics_bar.png",dpi=300); plt.close()
    
    # fp / fn grids
    save_grid(df_val,a.image_root,'FP',f"{a.out}/false_pos_grid.png",top_n=9,largest=True)
    save_grid(df_val,a.image_root,'FN',f"{a.out}/false_neg_grid.png",top_n=9,largest=False)
    print("Saved all visualisations to",a.out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',      required=True)
    ap.add_argument('--val-csv',    required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--out',        default='outputs/report_figs')
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--backbone',   default=BACKBONE)
    ap.add_argument('--dropout',    type=float, default=DROPOUT)
    main(ap.parse_args())
