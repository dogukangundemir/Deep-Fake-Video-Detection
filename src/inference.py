import argparse, pandas as pd, torch, os
from torch.utils.data import DataLoader
from dataset import DFDCFrames
from model   import build_model
from config  import *

def main(a):
    ds = DFDCFrames(a.meta_test, a.frames_dir+"/test", "test")
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    dev=torch.device(a.device)
    net=build_model(a.backbone, a.dropout); net.load_state_dict(torch.load(a.model, map_location=dev))
    net.to(dev).eval()
    frame_preds, vids = [], []
    with torch.no_grad():
        for x,vid in dl:
            p=torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().tolist()
            frame_preds+=p; vids+=vid
    df=pd.DataFrame({'video_id':vids,'pred':frame_preds})
    sub=pd.read_csv(a.sample_sub)
    sub['label']=sub.filename.str.replace('.mp4','').map(df.groupby('video_id').pred.mean()).fillna(0)
    sub.to_csv(a.out, index=False); print("Saved",a.out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--frames-dir', default='data/frames')
    p.add_argument('--meta-test',  required=True)
    p.add_argument('--sample-sub', required=True)
    p.add_argument('--out',        default='submission.csv')
    p.add_argument('--device',     default='cuda')
    p.add_argument('--backbone',   default=BACKBONE)
    p.add_argument('--dropout',    type=float, default=DROPOUT_P)
    main(p.parse_args())
