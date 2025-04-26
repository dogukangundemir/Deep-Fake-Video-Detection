# Deep-Fake-Video-Detection

# DFDC Subset Detector

**Setup**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 1  Prepare faces
python src/pipeline.py `
  --train-dir  data\train_sample_videos `
  --test-dir   data\test_videos `
  --sample-sub data\sample_submission.csv `
  --out-frames data\frames `
  --meta-train metadata_train.csv `
  --meta-val   metadata_val.csv `
  --meta-test  metadata_test.csv

### 2 Train 
python src/train.py --train-csv metadata_train.csv --val-csv metadata_val.csv --frames-dir data\frames

### 3 Evaluate & visualise FP/FN
python src/evaluate.py --model outputs\checkpoints\best.pt --csv metadata_val.csv
python src\test_viz.py  --model outputs\checkpoints\best.pt --meta-csv metadata_val.csv --frames-dir data\frames --dropout 0.3

#### 4 Create Kaggle submission
python src\inference.py --model outputs\checkpoints\best.pt --frames-dir data\frames --meta-test metadata_test.csv --sample-sub data\sample_submission.csv --out submission.csv

