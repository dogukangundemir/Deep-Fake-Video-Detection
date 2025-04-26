import random, numpy as np, torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_probs, thresh=0.5):
    y_pred = (y_probs >= thresh).astype(int)
    return {
        'auc': roc_auc_score(y_true, y_probs),
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def plot_confusion_matrix(cm, out_file):
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig(out_file)
    plt.close()