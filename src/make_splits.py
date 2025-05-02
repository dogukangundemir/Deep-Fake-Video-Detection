

import os, glob, argparse, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from config import SEED

def normalise_id(x:str)->str: return os.path.splitext(x)[0]

def frame_rows(label_file, image_root):
    df_vid = pd.read_csv(label_file)
    id_col = 'folder' if 'folder' in df_vid.columns else 'filename'
    df_vid['video_id'] = df_vid[id_col].apply(normalise_id)
    df_vid['label']    = df_vid['label'].map({'REAL':0,'FAKE':1}).astype(int)

    rows=[]
    for vid,lab in df_vid[['video_id','label']].itertuples(index=False):
        for jpg in glob.glob(os.path.join(image_root, vid, "*.jpg")):
            frame=os.path.splitext(os.path.basename(jpg))[0]
            rows.append({'video_id':vid,'frame':frame,'label':lab})
    return pd.DataFrame(rows).sort_values(['video_id','frame'])

def main(a):
    df=frame_rows(a.label_file,a.image_root)
    df.to_csv("metadata_all.csv",index=False)

    sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    folds=list(sgkf.split(df,df.label,df.video_id))
    tr_idx , val_idx , test_idx = folds[0][0] , folds[0][1] , folds[1][1]

    df.iloc[tr_idx ].to_csv("metadata_train.csv", index=False)
    df.iloc[val_idx].to_csv("metadata_val.csv"  , index=False)
    df.iloc[test_idx].to_csv("metadata_test.csv" , index=False)
    print("train:",len(tr_idx),"val:",len(val_idx),"test:",len(test_idx))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--label-file', required=True)
    ap.add_argument('--image-root', required=True)
    main(ap.parse_args())
