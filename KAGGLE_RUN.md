# 🚀 Kaggle Quick Start - RDFNet Task-Guided Training

## Repository: https://github.com/JMNuhin/RDF.git

---

## Step 1: Create Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Settings → **Accelerator: GPU T4 x2** (or available GPU)
4. Settings → **Internet: ON** ✅ (required!)
5. Settings → **Persistence: Files only**

---

## Step 2: Clone Repository

**Cell 1:**
```python
# Clone your GitHub repository
!git clone https://github.com/JMNuhin/RDF.git
%cd RDF
!ls -la
```

Expected output:
```
Cloning into 'RDF'...
total 53 files
✅ nets/
✅ utils/
✅ kaggle_train_task.py
✅ KAGGLE_SETUP.md
```

---

## Step 3: Add Datasets

Click **"Add data"** in Kaggle notebook and search for:

1. **VOC2007_FOG** - Your foggy training images
2. **VOC2007_Annotations** - PASCAL VOC 2007 annotations
3. *(Optional)* VOC2012_FOG + VOC2012_Annotations for more data
4. *(Optional)* Pretrained RDFNet checkpoint

Datasets will be at: `/kaggle/input/dataset-name/`

---

## Step 4: Verify Installation

**Cell 2:**
```python
# Check environment
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Verify files
!python -c "from nets.task_modules import SeverityHead; print('✅ Task modules loaded')"
!python -c "from utils.task_losses import InfoNCEContrastiveLoss; print('✅ Task losses loaded')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
GPU: Tesla T4
✅ Task modules loaded
✅ Task losses loaded
```

---

## Step 5: Prepare Dataset Annotations

**Option A: Generate from VOC dataset**

**Cell 3:**
```python
# Generate annotation files from VOC structure
from prepare_annotations import generate_annotations

VOC2007_path = '/kaggle/input/voc2007-fog/VOCdevkit/VOC2007'
# Adjust path based on your dataset name in Kaggle

generate_annotations(VOC2007_path, output_dir='./')
!head -5 2007_train.txt  # Preview
```

**Option B: Use existing annotation files**

**Cell 3:**
```python
# If you already have annotation TXT files, copy them
!cp /kaggle/input/your-annotations/2007_train.txt ./
!cp /kaggle/input/your-annotations/2007_val.txt ./
!head -5 2007_train.txt  # Preview
```

---

## Step 6: (Optional) Load Pretrained Weights

**Cell 4:**
```python
# If you have pretrained RDFNet weights
!mkdir -p model_data
!cp /kaggle/input/rdfnet-checkpoint/RDFNet.pth model_data/

# OR download from URL
# !wget YOUR_WEIGHTS_URL -O model_data/RDFNet.pth

!ls -lh model_data/*.pth
```

**Skip this if training from scratch** - will use YOLOv7-tiny backbone (already in repo).

---

## Step 7: Configure Training (Optional)

**Cell 5:**
```python
# View current configuration
!grep -A5 "# Training Configuration" kaggle_train_task.py | head -20

# Key settings you can modify:
# USE_TASK_HEADS = True      # Enable task-guided learning
# USE_DUAL_FOG = True         # Enable contrastive learning
# LAMBDA_MIN = 0.05           # Min adaptive λ
# LAMBDA_MAX = 0.20           # Max adaptive λ
# BETA = 0.1                  # Contrastive loss weight
# TEMPERATURE = 0.2           # InfoNCE temperature
```

**To modify:** Edit [kaggle_train_task.py](kaggle_train_task.py) before training, or for quick changes:

```python
# Quick config override (add to Cell 6 before training)
import sys
sys.argv = [
    'kaggle_train_task.py',
    '--batch-size', '8',  # Adjust if memory issues
]
```

---

## Step 8: Start Training! 🎯

**Cell 6:**
```python
# Train with task-guided contrastive learning
!python kaggle_train_task.py
```

**Expected output:**
```
🚀 RDFNet Training (Task-Guided)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Device: cuda
Task Heads: ✅ Enabled
Dual Fog Views: ✅ Enabled
λ adaptive: [0.05, 0.20], β: 0.1, τ: 0.2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Loading checkpoint: model_data/RDFNet.pth
✅ Loaded: 247 keys
⚠️  Skipped: 12 keys (new task modules)
   ➡️  severity_head.* (6 keys)
   ➡️  spatial_weight_heads.* (6 keys)
   
🔧 Task heads initialized randomly (warmup for 20 epochs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 1/300 - Warmup Phase
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
100%|██████████| 450/450 [03:24<00:00,  2.20it/s]

Epoch: 1/300
loss: 3.456 | det: 2.123 | dehazy: 0.543 | align: 0.432 | contr: 0.358 | λ: 0.127 | lr: 0.0010

Saving checkpoint: logs/ep001-loss3.456.pth
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**New metrics to watch:**
- `align` - Feature alignment loss (should decrease)
- `contr` - Contrastive loss (should decrease)
- `λ` - Adaptive weight (should vary 0.05-0.20)

---

## Step 9: Monitor Progress

**Cell 7 (run in parallel or after epoch completes):**
```python
# View recent checkpoints
!ls -lht logs/*.pth | head -5

# Check training log
!tail -50 logs/loss_*

# View latest metrics
import pandas as pd
import matplotlib.pyplot as plt

# If loss history saved as CSV/txt
# df = pd.read_csv('logs/loss_history.txt')
# df.plot(y=['det', 'align', 'contr'], figsize=(12,4))
# plt.show()
```

---

## Step 10: Save Results

Kaggle auto-saves to `/kaggle/working/`. Before session ends:

**Cell 8:**
```python
# Copy important checkpoints to output directory
!mkdir -p /kaggle/working/saved_checkpoints
!cp logs/best_epoch_weights.pth /kaggle/working/saved_checkpoints/
!cp logs/last_epoch_weights.pth /kaggle/working/saved_checkpoints/
!cp logs/ep*-loss*.pth /kaggle/working/saved_checkpoints/ 2>/dev/null || true

# Copy training logs
!cp -r logs/*.txt /kaggle/working/saved_checkpoints/ 2>/dev/null || true

!ls -lh /kaggle/working/saved_checkpoints/
print("\n✅ Files saved! Download from: Notebook → Output tab (after session ends)")
```

---

## 🔄 Resume Training (If Interrupted)

If Kaggle session times out, create new notebook:

**Resume Cell:**
```python
# Clone repo again
!git clone https://github.com/JMNuhin/RDF.git
%cd RDF

# Upload your previous checkpoint as Kaggle dataset first!
# Then copy it:
!mkdir -p logs
!cp /kaggle/input/my-checkpoint/ep050-loss2.456.pth logs/

# Training will auto-resume from latest epoch
!python kaggle_train_task.py
```

---

## 📊 Training Timeline (Estimated)

| Phase | Epochs | Duration (T4 x2) | Status |
|-------|--------|------------------|--------|
| **Warmup** | 0-20 | ~2 hours | New task heads learn |
| **Freeze Backbone** | 21-100 | ~8 hours | Heads + detection train |
| **Unfreeze All** | 101-300 | ~20 hours | Full fine-tuning |
| **Total** | **300** | **~30 hours** | May vary |

💡 **Tip:** Save checkpoints every 10-20 epochs. Kaggle GPU limit: 12h → restart session with resume!

---

## ⚙️ Configuration Reference

### **Full Task-Guided Training (Default)**
```python
USE_TASK_HEADS = True
USE_DUAL_FOG = True
LAMBDA_MIN = 0.05
LAMBDA_MAX = 0.20
BETA = 0.1
TEMPERATURE = 0.2
```

### **Ablation: Baseline (No Task Heads)**
Edit `kaggle_train_task.py`:
```python
USE_TASK_HEADS = False
USE_DUAL_FOG = False
```
Then train: creates baseline for comparison.

### **Ablation: Without Contrastive**
```python
USE_TASK_HEADS = True
USE_DUAL_FOG = False  # Only alignment, no InfoNCE
```

---

## 🐛 Troubleshooting

### **"No module named 'nets.task_modules'"**
```python
# Ensure you're in the RDF directory
%cd /kaggle/working/RDF
!ls nets/task_modules.py  # Should exist
```

### **"FileNotFoundError: 2007_train.txt"**
```python
# Generate annotations first (Step 5)
!ls *.txt  # Check if annotation files exist
```

### **"CUDA out of memory"**
Edit `kaggle_train_task.py`, reduce batch size:
```python
# Line ~95
batch_size = 4  # was 8
```

### **"λ_avg always constant (not varying)"**
- Check epoch ≥ 20 (warmup period)
- Severity head needs time to learn
- Verify task heads loaded: look for "Skipped: 12 keys" in output

### **"Training extremely slow"**
- Dual fog mode uses 3x batch size internally
- GPU T4 x2: ~2.2 it/s is normal
- Consider reducing batch_size to 4

---

## ✅ Quick Checklist

- [ ] GPU enabled in Kaggle notebook
- [ ] Internet ON
- [ ] Repository cloned: `git clone https://github.com/JMNuhin/RDF.git`
- [ ] VOC datasets added
- [ ] Annotation files generated/copied
- [ ] (Optional) Pretrained weights added
- [ ] Training started: `!python kaggle_train_task.py`
- [ ] Monitoring new metrics: align, contr, λ
- [ ] Checkpoints saving to `/kaggle/working/`

---

## 📈 Expected Results

After full 300-epoch training:
- **Detection mAP**: ~70-75% (on VOC-FOG val)
- **Real fog (RTTS)**: 10-15% improvement over baseline
- **Alignment loss**: < 0.1 (from ~0.4)
- **Contrastive loss**: < 0.05 (from ~0.3)
- **λ variation**: 0.05-0.20 (severity-adaptive)

---

## 🎯 After Training

1. **Download checkpoints** from Output tab
2. **Evaluate** on RTTS test set: `!python kaggle_eval.py`
3. **Compute mAP**: `!python get_map.py`
4. **Visualize** detections: `!python predict.py`
5. **Compare** task-guided vs baseline (ablation)

---

## 📚 Documentation

- [TASK_GUIDED_README.md](TASK_GUIDED_README.md) - Full technical details
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Change log
- [QUICK_START.md](QUICK_START.md) - Local training guide

---

**Ready to train! Copy-paste cells into Kaggle notebook and run sequentially.** 🚀

**Questions?** Check troubleshooting section or detailed docs.
