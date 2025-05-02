
import matplotlib.pyplot as plt
import os
from glob import glob

def show_pair(raw, crop):
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(raw[...,::-1])  
    axes[0].set_title("Raw Frame")
    axes[1].imshow(crop[...,::-1])
    axes[1].set_title("MTCNN Crop")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sub = os.path.join('data','frames','train', os.listdir('data/frames/train')[0])
    raws  = sorted(glob(os.path.join(sub, 'raw_*.jpg')))
    crops = sorted(glob(os.path.join(sub, 'crop_*.jpg')))
    if not raws or not crops:
        print("No raws or crops found in", sub)
    else:
        import cv2
        raw = cv2.imread(raws[0])
        crop = cv2.imread(crops[0])
        show_pair(raw, crop)
