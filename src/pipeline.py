
import argparse, json, os, random, cv2, numpy as np, pandas as pd, torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from sklearn.model_selection import StratifiedGroupKFold
from config import *

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

def sample_frames(total, n): return np.linspace(0, total-1, n, dtype=int)
def get_device(): return 'cuda' if torch.cuda.is_available() else 'cpu'

def crop_video(mp4, out_dir, mtcnn):
    cap = cv2.VideoCapture(mp4); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in sample_frames(total, NUM_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, frame = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is None: continue
        arr  = (face.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        crop = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out  = f"{out_dir}/crop_{idx}.jpg"
        cv2.imwrite(out, crop); frames.append(idx)
    cap.release(); return frames

def main(a):
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, device=get_device())
    os.makedirs(a.out_frames, exist_ok=True)

    # TRAIN
    meta_path = os.path.join(a.train_dir, "metadata.json")
    train_meta = json.load(open(meta_path))
    rows = []
    for fn, info in tqdm(train_meta.items(), desc="Cropping train"):
        vid      = fn[:-4]
        label    = int(info['label'] == 'FAKE')
        vid_dir  = f"{a.out_frames}/train/{vid}"
        os.makedirs(vid_dir, exist_ok=True)
        frames   = crop_video(f"{a.train_dir}/{fn}", vid_dir, mtcnn)
        rows    += [{'video_id':vid,'frame':f,'label':label} for f in frames]

    df = pd.DataFrame(rows)
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(gkf.split(df, df.label, df.video_id))
    df.iloc[train_idx].to_csv(a.meta_train, index=False)
    df.iloc[val_idx]  .to_csv(a.meta_val,   index=False)

    # TEST
    test_fns  = [l.split(',')[0] for l in open(a.sample_sub).read().splitlines()[1:]]
    rows = []
    for fn in tqdm(test_fns, desc="Cropping test"):
        vid_dir = f"{a.out_frames}/test/{fn[:-4]}"; os.makedirs(vid_dir, exist_ok=True)
        frames  = crop_video(f"{a.test_dir}/{fn}", vid_dir, mtcnn)
        rows   += [{'video_id':fn[:-4],'frame':f} for f in frames]
    pd.DataFrame(rows).to_csv(a.meta_test, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir',  required=True)
    p.add_argument('--test-dir',   required=True)
    p.add_argument('--sample-sub', required=True)
    p.add_argument('--out-frames', required=True)
    p.add_argument('--meta-train', required=True)
    p.add_argument('--meta-val',   required=True)
    p.add_argument('--meta-test',  required=True)
    main(p.parse_args())
