
**Deep-Fake Video Detection\
End-to-End Frames-Only Pipeline**


# 1 Repository Layout 

    Deep-Fake-Video-Detection/
    │
    ├─ archive/
    │   └─ images/               ← one sub-folder per video / subject
    │        ├─ zsrddprmz/       ← frame_#.jpg   (already MTCNN-cropped)
    │        └─ ...
    │
    ├─ metadata34.csv            ← Kaggle video-level sheet: filename,label,...
    │
    ├─ src/
    │   config.py                ← hyper-parameters (backbone, dropout …)
    │   make_splits.py           ← writes train / val / test CSVs
    │   train.py                 ← freeze→head-tune→unfreeze training loop
    │   viz.py                   ← 6-figure validation dashboard
    │   test_eval.py             ← 8-figure test dashboard (adds calibration)
    │   inference.py             ← optional CSV for unlabeled Kaggle test
    │   … (augment.py, dataset.py, model.py, …)
    │
    ├─ requirements.txt
    └─ outputs/                  ← created automatically (checkpoints, figs)

# 2 Quick Start (Windows PowerShell) 

1.  **Create and activate venv**

        PS> python -m venv venv
        PS> .\venv\Scripts\Activate

2.  **Install dependencies**\

        PS> pip install --upgrade pip
        PS> pip install --index-url https://download.pytorch.org/whl/cu118 ^
              torch==2.2.2 torchvision==0.17.2
        PS> pip install -r requirements.txt --no-deps

3.  **Build frame-level CSVs & train/val/test split**\

        PS> python src\make_splits.py ^
              --label-file metadata34.csv ^
              --image-root archive\images
        # ➜ metadata_train.csv / metadata_val.csv / metadata_test.csv

4.  **Train EfficientNet-B0 (freeze 3 epochs ⇒ unfreeze)**\

        PS> python src\train.py ^
              --train-csv  metadata_train.csv ^
              --val-csv    metadata_val.csv ^
              --image-root archive\images
        # outputs/checkpoints/best.pt

5.  **Validation dashboard (6 PNGs)**\

        PS> python src\viz.py ^
              --model      outputs\checkpoints\best.pt ^
              --val-csv    metadata_val.csv ^
              --image-root archive\images ^
              --out        outputs\val_figs ^
              --dropout    0.3

6.  **Test dashboard (8 PNGs)**\

        PS> python src\test_eval.py ^
              --model     outputs\checkpoints\best.pt ^
              --test-csv  metadata_test.csv ^
              --image-root archive\images ^
              --out       outputs\test_figs ^
              --dropout   0.3

# 3 Output Artifacts 

-   `outputs/checkpoints/best.pt` -- fully fine-tuned weights.

-   `outputs/tb/` -- TensorBoard learning curves.

-   `outputs/val_figs/` -- *roc_curve.png*, *pr_curve.png*,
    *cmatrix.png*, *metrics_bar.png*, *false_pos_grid.png*,
    *false_neg_grid.png*.

-   `outputs/test_figs/` -- validation set figures *plus*
    *calib_curve.png*, *score_hist.png*.

# 4 Key Implementation Details

::: description
Already-cropped faces $(224\times224)$ from DFDC; stratified *group*
split prevents subject leakage.

Any timm / torchvision backbone (`BACKBONE` in `config.py`). Default:
EfficientNet-B0.

First `FREEZE_EPOCHS=3` epochs: backbone frozen; only head learns. Then
backbone unfrozen at lower LR (`UNFROZEN_LR`).

MixUp $\alpha=0.4$ **and** label-smoothing $\varepsilon=0.05$ + Dropout
0.3.

TensorBoard (always) and optional Weights&Biases (`wandb.init()` in
`train.py`).

