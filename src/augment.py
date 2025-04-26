import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE
def train_aug(): return A.Compose([
    A.HorizontalFlip(0.5), A.ShiftScaleRotate(0.1,0.1,10,p=0.5),
    A.ColorJitter(0.2,0.2,0.2,0.1,p=0.5), A.Resize(IMG_SIZE,IMG_SIZE), ToTensorV2()
])
def val_aug(): return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), ToTensorV2()])
