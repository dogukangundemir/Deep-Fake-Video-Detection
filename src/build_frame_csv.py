
import os, glob, argparse, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def normalise_id(s: str) -> str:
    return os.path.splitext(s)[0]

def main(a):
    df_vid = pd.read_csv(a.label_file)

    # detect which column holds the ID
    if 'folder' in df_vid.columns:
        id_col = 'folder'
    elif 'filename' in df_vid.columns:
        id_col = 'filename'
    else:
        raise ValueError("CSV must contain a 'folder' or 'filename' column.")

    df_vid['video_id'] = df_vid[id_col].apply(normalise_id)
    df_vid['label'] = df_vid['label'].map({'REAL':0, 'FAKE':1}).astype(int)

    rows = []
    for vid, lab in df_vid[['video_id', 'label']].itertuples(index=False):
        img_dir = os.path.join(a.image_root, vid)
        for jpg in glob.glob(os.path.join(img_dir, "*.jpg")):
            frame = os.path.splitext(os.path.basename(jpg))[0]  # 0, 30, 60 …
            rows.append({'video_id': vid, 'frame': frame, 'label': lab})

    df = pd.DataFrame(rows).sort_values(['video_id', 'frame'])
    df.to_csv(a.out_all, index=False)
    print(f"{a.out_all}  | rows: {len(df)}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, vl = next(sgkf.split(df, df.label, df.video_id))
    df.iloc[tr].to_csv(a.out_train, index=False); print("train :", len(tr))
    df.iloc[vl].to_csv(a.out_val,   index=False); print("val   :", len(vl))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--label-file', required=True)      
    ap.add_argument('--image-root', required=True)     
    ap.add_argument('--out-all',    default='metadata_all.csv')
    ap.add_argument('--out-train',  default='metadata_train.csv')
    ap.add_argument('--out-val',    default='metadata_val.csv')
    main(ap.parse_args())
