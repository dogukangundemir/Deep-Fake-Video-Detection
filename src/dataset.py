import os, cv2, torch, pandas as pd
from torch.utils.data import Dataset
from augment import train_aug, val_aug

class DFDCFrames(Dataset):
    def __init__(self, csv, image_root, mode):
        self.df = pd.read_csv(csv)
        self.image_root=image_root
        self.mode=mode
        self.aug = train_aug() if mode=="train" else val_aug()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row.video_id, f"{row.frame}.jpg")
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img  = self.aug(image=img)['image']/255.
        if self.mode=="test": return img, row.video_id
        return img, torch.tensor(row.label, dtype=torch.float32)
