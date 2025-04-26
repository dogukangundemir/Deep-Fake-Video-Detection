import cv2, torch, pandas as pd, os
from torch.utils.data import Dataset
from augment import train_aug, val_aug

class DFDCFrames(Dataset):
    def __init__(self, csv, frames_dir, mode):
        self.df = pd.read_csv(csv); self.frames_dir = frames_dir
        self.mode = mode; self.aug = train_aug() if mode=="train" else val_aug()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(f"{self.frames_dir}/{row.video_id}/crop_{row.frame}.jpg"),
                           cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)['image']/255.
        if self.mode=="test": return img, row.video_id
        return img, torch.tensor(row.label, dtype=torch.float32)
