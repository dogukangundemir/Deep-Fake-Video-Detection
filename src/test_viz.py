# ────────────────────────────────────────────────────────────────
# src/test_viz.py   (DROP-IN REPLACEMENT)
# ────────────────────────────────────────────────────────────────
import os, argparse, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset import FakeFrameDataset
from model   import create_model
from sklearn.model_selection import StratifiedGroupKFold

# ── helpers ─────────────────────────────────────────────────────
def infer_on_fold(meta_csv, frames_dir, model_path, backbone, dropout, device):
    df = pd.read_csv(meta_csv)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, test_idx = list(sgkf.split(df, df.label, df.video_id))[-1]

    ds = FakeFrameDataset(meta_csv, frames_dir, mode='train')
    loader = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False)

    dev = torch.device(device)
    net = create_model(backbone, pretrained=False, dropout=dropout)
    net.load_state_dict(torch.load(model_path, map_location=dev), strict=False)
    net.to(dev).eval()

    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += net(x.to(dev)).squeeze(1).cpu().sigmoid().tolist()
            truths += y.tolist()

    df_test = df.iloc[test_idx].copy()
    df_test['pred'] = preds
    df_test['true'] = truths
    return df_test

def plot_roc(df, path):
    fpr, tpr, _ = roc_curve(df.true, df.pred)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC (held-out fold)'); plt.legend(); plt.tight_layout()
    plt.savefig(path); plt.close()

def plot_cm(df, path):
    cm = confusion_matrix(df.true, df.pred>0.5, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Real','Fake'])
    fig, ax = plt.subplots(); disp.plot(ax=ax); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_examples(df, frames_dir, out_path, tag, top_n=3, largest=True):
    sub = (df[df.true==(0 if tag=='FP' else 1)]
           .sort_values('pred', ascending=not largest).head(top_n))
    plt.figure(figsize=(top_n*3,3))
    for i,row in enumerate(sub.itertuples()):
        img = plt.imread(os.path.join(frames_dir, row.video_id, f"crop_{int(row.frame)}.jpg"))
        ax = plt.subplot(1, top_n, i+1); ax.imshow(img); ax.axis('off')
        ax.set_title(f"{tag}\np={row.pred:.2f}")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# ── main entry ──────────────────────────────────────────────────
def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    df = infer_on_fold(a.meta_csv, a.frames_dir, a.model, a.backbone, a.dropout, a.device)

    plot_roc(df, os.path.join(a.out_dir,'roc_test.png'))
    plot_cm(df,  os.path.join(a.out_dir,'cm_test.png'))
    plot_examples(df, a.frames_dir, os.path.join(a.out_dir,'fp_test.png'), 'FP', a.num_samples, largest=True)
    plot_examples(df, a.frames_dir, os.path.join(a.out_dir,'fn_test.png'), 'FN', a.num_samples, largest=False)

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    print(f"AUC {roc_auc_score(df.true,df.pred):.4f}  "
          f"ACC {accuracy_score(df.true,df.pred>0.5):.4f}  "
          f"F1 {f1_score(df.true,df.pred>0.5):.4f}")
    print("Plots saved to", a.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--meta-csv',    required=True)
    p.add_argument('--frames-dir',  required=True)
    p.add_argument('--model',       required=True)
    p.add_argument('--backbone',    default='efficientnet_b0')
    p.add_argument('--dropout',     type=float, default=0.0)   # <-- NOW ACCEPTED
    p.add_argument('--device',      default='cuda')
    p.add_argument('--out-dir',     default='outputs/test_viz')
    p.add_argument('--num-samples', type=int, default=3)
    args = p.parse_args(); main(args)
# ────────────────────────────────────────────────────────────────
