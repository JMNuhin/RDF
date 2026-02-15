# 🚀 Kaggle Setup Guide - RDFNet Task-Guided Training

## Quick Start in Kaggle

### **Step 1: Create a New Kaggle Notebook**

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Settings → **GPU T4 x2** (or available GPU)
4. Settings → **Internet ON** (required for git clone)

---

### **Step 2: Clone Repository**

```python
# Cell 1: Clone the repository
!git clone https://github.com/YOUR_USERNAME/RDF_net-main.git
%cd RDF_net-main
!ls -la
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

---

### **Step 3: Add Required Kaggle Datasets**

In Kaggle Notebook → **Add Data** → Search and add:

1. **VOC2007_FOG** (your fog dataset)
2. **VOC2007_Annotations** (PASCAL VOC 2007)
3. **VOC2012_FOG** (optional, for more training data)
4. **VOC2012_Annotations** (optional)
5. **RDFNet-pretrained** (if you have pretrained weights)

Datasets will be available at: `/kaggle/input/dataset-name/`

---

### **Step 4: Install Dependencies (if needed)**

```python
# Cell 2: Check PyTorch version
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Most packages are pre-installed in Kaggle
# If you need additional packages:
# !pip install package-name
```

---

### **Step 5: Prepare Dataset Paths**

Create a cell to generate annotation files:

```python
# Cell 3: Generate training annotations
from prepare_annotations import generate_annotations

# Adjust paths to Kaggle input directories
VOC2007_path = '/kaggle/input/voc2007-fog/VOCdevkit/VOC2007'
VOC2012_path = '/kaggle/input/voc2012-fog/VOCdevkit/VOC2012'  # if available

# Generate annotation files
generate_annotations(VOC2007_path, output_dir='./')
print("✅ Annotation files created")
```

**Or manually copy your existing annotation files:**

```python
# If you have pre-generated 2007_train.txt and 2007_val.txt
!cp /kaggle/input/your-annotations-dataset/2007_train.txt ./
!cp /kaggle/input/your-annotations-dataset/2007_val.txt ./
```

---

### **Step 6: Configure Training**

Edit [kaggle_train_task.py](kaggle_train_task.py) paths if needed:

```python
# Cell 4: Quick configuration check
!grep -n "train_annotation_path" kaggle_train_task.py
!grep -n "val_annotation_path" kaggle_train_task.py

# Should point to:
# train_annotation_path = "2007_train.txt"
# val_annotation_path = "2007_val.txt"
```

---

### **Step 7: Download Pretrained Weights (Optional)**

If starting from pretrained RDFNet:

```python
# Cell 5: Copy pretrained weights
!mkdir -p model_data
!cp /kaggle/input/rdfnet-pretrained/RDFNet.pth model_data/
# OR download from URL
# !wget https://your-weights-url.com/RDFNet.pth -P model_data/
```

If starting from scratch:
- The code will use `yolov7_tiny_weights.pth` (should be in repo)
- Or download YOLOv7-tiny backbone if needed

---

### **Step 8: Start Training! 🎯**

#### **Option A: Task-Guided Training (New Method)**

```python
# Cell 6: Train with task-guided contrastive learning
!python kaggle_train_task.py
```

**Expected output:**
```
🚀 RDFNet Training (Task-Guided)
Device: cuda
Task Heads: ✅ Enabled
Dual Fog Views: ✅ Enabled
λ adaptive: [0.05, 0.20], β: 0.1, τ: 0.2

Epoch: 1/300
loss: 3.456 | loss_det: 2.123 | dehazy: 0.543 | align: 0.432 | contr: 0.358 | λ: 0.127 | lr: 0.001
```

#### **Option B: Baseline Training (Original RDFNet)**

```python
# For ablation/comparison
!python kaggle_train.py
```

---

### **Step 9: Monitor Training**

```python
# Cell 7: Check training progress
!ls -lh logs/
!tail -n 20 logs/loss_*.txt  # View recent losses
```

**View loss curves:**
```python
# Cell 8: Plot training curves
from utils.callbacks import LossHistory
import matplotlib.pyplot as plt

# Load and plot (adjust paths as needed)
# ... plotting code ...
```

---

### **Step 10: Save Results**

Kaggle auto-saves outputs in `/kaggle/working/`. To keep your checkpoints:

```python
# Cell 9: Copy important files before session ends
!mkdir -p /kaggle/working/saved_models
!cp logs/best_epoch_weights.pth /kaggle/working/saved_models/
!cp logs/last_epoch_weights.pth /kaggle/working/saved_models/
!cp logs/loss_*.txt /kaggle/working/saved_models/

print("✅ Models saved to /kaggle/working/saved_models/")
print("📥 Download from: Notebook → Output tab (after session ends)")
```

---

## 🎛️ Configuration Options

### **Task-Guided Training Flags**

Edit in [kaggle_train_task.py](kaggle_train_task.py):

```python
# Disable task heads for baseline
USE_TASK_HEADS = False

# Disable contrastive learning
USE_DUAL_FOG = False

# Adjust hyperparameters
LAMBDA_MIN = 0.05    # Min adaptive λ
LAMBDA_MAX = 0.20    # Max adaptive λ
BETA = 0.1           # Contrastive weight
TEMPERATURE = 0.2    # InfoNCE temperature
```

---

## 🔄 Resume Training

If training stops, clone repo again and resume:

```python
# Load previous checkpoint
!git clone https://github.com/YOUR_USERNAME/RDF_net-main.git
%cd RDF_net-main

# Copy checkpoint from previous session
# (upload as Kaggle dataset first)
!cp /kaggle/input/my-checkpoint/ep050-loss2.456.pth logs/

# Training will auto-resume from latest epoch
!python kaggle_train_task.py
```

---

## 📊 Expected Timeline

| Phase | Epochs | Time (T4 x2) | Notes |
|-------|--------|--------------|-------|
| Warmup | 0-20 | ~2 hours | Task heads learning |
| Freeze | 21-100 | ~8 hours | Backbone frozen |
| Unfreeze | 101-300 | ~20 hours | Full fine-tuning |
| **Total** | **300** | **~30 hours** | May vary |

**Tips:**
- Save checkpoints every 10 epochs
- Kaggle sessions: 12h GPU limit (restart if needed)
- Use Kaggle Datasets to persist checkpoints between sessions

---

## 🧪 Validation

After training:

```python
# Evaluate on test set
!python kaggle_eval.py

# Check mAP
!python get_map.py
```

---

## 🐛 Troubleshooting

### **"No module named 'nets.task_modules'"**
```python
# Ensure you're in the repository directory
%cd /kaggle/working/RDF_net-main
!ls nets/task_modules.py  # Should exist
```

### **"Checkpoint not found"**
```python
# Check logs directory
!ls -la logs/
# If empty, training starts from scratch (expected first run)
```

### **"CUDA out of memory"**
```python
# Reduce batch size in kaggle_train_task.py
# Edit line: batch_size = 8  # Try 4 or 2
```

### **"Different fog intensities but λ_avg constant"**
```python
# Severity head not learning - check:
# 1. Task heads initialized? (should see "Skipped: 12 keys")
# 2. Warmup complete? (wait until epoch 20+)
# 3. Learning rate for new heads? (should be 1e-3)
```

---

## 📁 Repository Structure After Clone

```
RDF_net-main/
├── kaggle_train_task.py ⭐ Main training script
├── kaggle_train.py       (baseline)
├── nets/
│   ├── model.py          (modified)
│   └── task_modules.py   ⭐ New task heads
├── utils/
│   ├── dataloader.py     (modified)
│   ├── task_losses.py    ⭐ New losses
│   └── utils_fit_task.py ⭐ New training loop
├── QUICK_START.md        (this guide)
├── TASK_GUIDED_README.md (detailed docs)
└── model_data/
    ├── yolo_anchors.txt
    └── rtts_classes.txt
```

---

## ✅ Quick Checklist

- [ ] Create Kaggle notebook with GPU
- [ ] Enable internet access
- [ ] Clone repository
- [ ] Add VOC datasets
- [ ] Generate/copy annotation files
- [ ] (Optional) Add pretrained weights
- [ ] Run `kaggle_train_task.py`
- [ ] Monitor logs for new metrics
- [ ] Save checkpoints to /kaggle/working/
- [ ] Download results after training

---

## 🎯 Next Steps After Training

1. **Download checkpoints** from Kaggle Output tab
2. **Evaluate** on RTTS real fog test set
3. **Compare** task-guided vs. baseline (ablation)
4. **Visualize** predictions with `predict.py`
5. **Fine-tune** hyperparameters if needed

---

**Happy Training! 🚀**

For detailed architecture and loss formulas, see [TASK_GUIDED_README.md](TASK_GUIDED_README.md).
