
import os, argparse, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (roc_curve,precision_recall_curve, average_precision_score,
                             confusion_matrix, accuracy_score, f1_score, roc_auc_score,)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from dataset import DFDCFrames
from model   import build_model
from config  import BACKBONE, DROPOUT, BATCH_SIZE

def save_grid(df, root, tag, path, n=9, largest=True):
    sub=(df[df.true==(0 if tag=='FP' else 1)]
         .sort_values('pred',ascending=not largest).head(n))
    cols=int(np.sqrt(n)); rows=int(np.ceil(n/cols))
    plt.figure(figsize=(cols*3,rows*3))
    for i,r in enumerate(sub.itertuples()):
        img=plt.imread(os.path.join(root,r.video_id,f"{r.frame}.jpg"))
        ax=plt.subplot(rows,cols,i+1); ax.imshow(img); ax.axis('off')
        ax.set_title(f"p={r.pred:.2f}")
    plt.suptitle(tag); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def main(a):
    os.makedirs(a.out,exist_ok=True)
    dev=torch.device(a.device)
    ds=DFDCFrames(a.test_csv,a.image_root,"val")
    dl=DataLoader(ds,BATCH_SIZE,shuffle=False)

    net=build_model(a.backbone,a.dropout)
    net.load_state_dict(torch.load(a.model,map_location=dev), strict=False)
    net.to(dev).eval()

    preds,truth=[],[]
    with torch.no_grad():
        for x,y in dl:
            preds+=torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist()
            truth+=y.tolist()
    df=pd.read_csv(a.test_csv); df['pred']=preds; df['true']=truth

    # 1—roc
    fpr,tpr,_=roc_curve(truth,preds); auc=roc_auc_score(truth,preds)
    plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC AUC={auc:.3f}")
    plt.savefig(f"{a.out}/roc_curve.png",dpi=300); plt.close()
    
    # 2—pr
    prec,rec,_=precision_recall_curve(truth,preds)
    ap=average_precision_score(truth,preds)
    plt.plot(rec,prec); plt.title(f"PR AP={ap:.3f}")
    plt.savefig(f"{a.out}/pr_curve.png",dpi=300); plt.close()
    
    # 3—confusion
    cm=confusion_matrix(truth,np.array(preds)>0.5)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues'); plt.title('Confusion matrix')
    plt.savefig(f"{a.out}/cmatrix.png",dpi=300); plt.close()
    
    # 4—metrics bar
    acc=accuracy_score(truth,np.array(preds)>0.5)
    f1 =f1_score(truth,np.array(preds)>0.5)
    sns.barplot(x=['AUC','ACC','F1'],y=[auc,acc,f1]); plt.ylim(0,1)
    plt.title('Score summary'); plt.savefig(f"{a.out}/metrics_bar.png",dpi=300); plt.close()
    
    # 5—calibration curve
    prob_true,prob_pred=calibration_curve(truth,preds,n_bins=10)
    plt.plot(prob_pred,prob_true,'o-'); plt.plot([0,1],[0,1],'--')
    plt.title('Calibration curve'); plt.xlabel('Predicted'); plt.ylabel('Empirical')
    plt.savefig(f"{a.out}/calib_curve.png",dpi=300); plt.close()
    
    # 6—score histogram
    plt.hist(preds,bins=30,alpha=0.7); plt.title('Histogram of probabilities')
    plt.savefig(f"{a.out}/score_hist.png",dpi=300); plt.close()
    # 7 & 8 — fp / fn grids
    save_grid(df,a.image_root,'FP',f"{a.out}/false_pos_grid.png")
    save_grid(df,a.image_root,'FN',f"{a.out}/false_neg_grid.png",largest=False)

    print("All testing visualisations saved to",a.out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',      required=True)
    ap.add_argument('--test-csv',   required=True)  
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--out',        default='outputs/test_figs')
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--backbone',   default=BACKBONE)
    ap.add_argument('--dropout',    type=float, default=DROPOUT)
    main(ap.parse_args())
