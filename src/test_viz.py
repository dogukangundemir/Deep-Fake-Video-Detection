import os, argparse, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve, auc ,roc_auc_score,precision_recall_curve, average_precision_score, \
                            confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from dataset import DFDCFrames                
from model   import build_model
from config  import BACKBONE, DROPOUT_P, BATCH_SIZE, SEED

def save_roc(y, p, path):
    fpr,tpr,_ = roc_curve(y,p); roc_auc = auc(fpr,tpr)
    plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC curve'); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def save_pr(y, p, path):
    prec,rec,_=precision_recall_curve(y,p)
    ap=average_precision_score(y,p)
    plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR curve')
    plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def save_cm(y, p, path):
    cm = confusion_matrix(y, np.array(p) > 0.5, labels=[0,1])
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title('Confusion matrix'); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def save_bar(auc_v,acc_v,f1_v,path):
    plt.figure(); sns.barplot(x=['AUC','ACC','F1'],y=[auc_v,acc_v,f1_v])
    plt.ylim(0,1); plt.title('Score summary'); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def save_grid(df, frames_dir, tag, path, top_n=9, largest=True):
    subset = (df[df.true==(0 if tag=='FP' else 1)]
              .sort_values('pred',ascending=not largest).head(top_n))
    cols = int(np.sqrt(top_n)); rows = int(np.ceil(top_n/cols))
    plt.figure(figsize=(cols*3, rows*3))
    for i,row in enumerate(subset.itertuples()):
        img = plt.imread(os.path.join(frames_dir,'train',row.video_id,f"crop_{int(row.frame)}.jpg"))
        ax=plt.subplot(rows,cols,i+1); ax.imshow(img); ax.axis('off')
        ax.set_title(f"p={row.pred:.2f}")
    plt.suptitle(f"{tag} samples"); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    
    df = pd.read_csv(a.meta_csv)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, test_idx = list(sgkf.split(df, df.label, df.video_id))[-1]
    ds = DFDCFrames(a.meta_csv, os.path.join(a.frames_dir,'train'), "val")
    dl = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    
    dev=torch.device(a.device)
    net=build_model(a.backbone, a.dropout); net.load_state_dict(torch.load(a.model,map_location=dev),strict=False)
    net.to(dev).eval()

   
    preds, y = [], []
    with torch.no_grad():
        for x,t in dl:
            preds += torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist()
            y     += t.tolist()
    df_test = df.iloc[test_idx].copy(); df_test['pred']=preds; df_test['true']=y

    
    save_roc(y,preds,f"{a.out_dir}/roc_curve.png")
    save_pr (y,preds,f"{a.out_dir}/pr_curve.png")
    save_cm (y,preds,f"{a.out_dir}/cmatrix.png")
    auc_val = roc_auc_score(y, preds)
    acc_val = accuracy_score(y, np.array(preds) > 0.5)
    f1_val  = f1_score     (y, np.array(preds) > 0.5)
    save_bar(auc_val, acc_val, f1_val, f"{a.out_dir}/metrics_bar.png")
    save_grid(df_test,a.frames_dir,'FP',f"{a.out_dir}/false_pos_grid.png",top_n=9,largest=True)
    save_grid(df_test,a.frames_dir,'FN',f"{a.out_dir}/false_neg_grid.png",top_n=9,largest=False)
    print("Saved all figures to", a.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',      required=True)
    ap.add_argument('--meta-csv',   required=True)
    ap.add_argument('--frames-dir', required=True)
    ap.add_argument('--backbone',   default=BACKBONE)
    ap.add_argument('--dropout',    type=float, default=DROPOUT_P)
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--out-dir',    default='outputs/test_viz')
    args = ap.parse_args(); run(args)