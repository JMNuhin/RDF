# Task-Guided Contrastive Learning for RDFNet

## Implementation Summary

This modification adds **task-guided contrastive learning** with **adaptive λ(x) weighting** and **spatially adaptive feature supervision** to RDFNet.

---

## 🎯 Key Features

### 1. **Severity-Adaptive Loss Weighting**
- `SeverityHead` predicts fog severity s(x) from backbone features
- Dynamic loss weight: λ(x) = λ_min + (λ_max - λ_min) * σ(s(x))  
- Automatically balances feature alignment based on fog density

### 2. **Spatially Adaptive Feature Alignment**
- `SpatialWeightHead` generates per-pixel weight maps (W3, W4, W5)
- Weighted MSE loss focuses on important spatial regions
- Multi-scale supervision (P3, P4, P5 neck features)

### 3. **Contrastive Feature Invariance**
- InfoNCE loss pulls together features from two fog views
- Multi-scale feature pooling for robust representation
- Temperature-controlled similarity metric

### 4. **Dual Fog View Generation**
- Data loader creates two views with different fog intensities
- Same geometric transforms, different fog parameters
- Enables self-supervised contrastive learning

---

## 📁 New/Modified Files

### **New Modules**
- `nets/task_modules.py` - SeverityHead, SpatialWeightHead, adaptive λ computation
- `utils/task_losses.py` - Feature alignment loss, InfoNCE contrastive loss
- `utils/utils_fit_task.py` - Updated training loop with task losses
- `kaggle_train_task.py` - Training script with task-guided learning

### **Modified Files**
- `nets/model.py` - Added task heads, returns neck features during training
- `utils/dataloader.py` - Supports dual fog view generation
- *(Original files preserved - new functionality is backward compatible)*

---

## 🚀 Usage

### **Option 1: Kaggle Training (Recommended)**

```python
# Use the new training script
!python kaggle_train_task.py
```

### **Option 2: Configuration**

Edit hyperparameters in `kaggle_train_task.py`:

```python
# Task-guided learning settings
USE_TASK_HEADS = True  # Enable/disable task heads
USE_DUAL_FOG = True  # Enable/disable dual fog views
LAMBDA_MIN = 0.05  # Minimum λ
LAMBDA_MAX = 0.20  # Maximum λ
BETA = 0.1  # Contrastive loss weight
TEMPERATURE = 0.2  # InfoNCE temperature
WARMUP_EPOCHS = 20  # Warmup for task heads
```

### **Option 3: Resume from Checkpoint**

The script automatically handles checkpoints:

```python
# Loads RDFNet.pth with strict=False
# New task head weights initialized randomly
# Backbone + detector weights loaded successfully
```

### **Option 4: Ablation Studies**

Disable specific components:

```python
# Disable task-guided learning (use baseline RDFNet)
USE_TASK_HEADS = False

# Use task heads but no contrastive (only alignment)
USE_DUAL_FOG = False

# Disable alignment during warmup
use_task_losses = False  # in training loop
```

---

## 🔬 Training Pipeline

### **Loss Function**

```
L_total = L_det + λ(x) * L_align + β * L_con
```

Where:
- **L_det**: YOLOv7 detection loss (cls + box + obj)
- **L_align**: Weighted MSE between hazy and clean features
- **L_con**: InfoNCE contrastive loss between two fog views
- **λ(x)**: Adaptive weight from severity prediction
- **β**: Fixed contrastive loss weight

### **Training Stages**

1. **Warmup (Epochs 0-20)**: Train task heads with frozen backbone
2. **Frozen (Epochs 21-100)**: Freeze backbone, train detector + task heads
3. **Unfreeze (Epochs 101-300)**: Full end-to-end training

### **Multi-Scale Feature Supervision**

```
Features:
┌──────┬─────────┬────────┬──────────┐
│Scale │ Res     │ Dims   │ Weight   │
├──────┼─────────┼────────┼──────────┤
│ P3   │ 80x80   │ 128    │ W3       │
│ P4   │ 40x40   │ 256    │ W4       │
│ P5   │ 20x20   │ 512    │ W5       │
└──────┴─────────┴────────┴──────────┘

L_align = Σ MSE(P_hazy * W, P_clean * W)
```

---

## 📊 Expected Outputs

### **Training Logs**

```
Epoch: 150/300
Total Loss: 2.145
  Detection: 1.523 | Dehazy: 0.382 | Align: 0.156 | Contrast: 0.084 | λ_avg: 0.132
```

### **Checkpoint Structure**

Saved checkpoints include:
- Backbone weights
- Detector weights (YOLO heads)
- Dehazing network (LMDNet)
- **Severity head** (new)
- **Spatial weight heads** (new)
- EMA weights

---

## 🛠️ Troubleshooting

### **Issue: "KeyError: detections"**
**Solution:** Model not in task-guided mode. Check `use_task_heads=True` in model init.

### **Issue: "num_samples=0"**
**Solution:** Annotation files not generated. Run annotation generation first.

### **Issue: "Shape mismatch in features"**
**Solution:** Ensure neck_features are cloned before downsample layers modify them.

### **Issue: "Checkpoint loading fails"**
**Solution:** Use `strict=False` when loading - new modules won't be in old checkpoints.

### **Issue: "Training crashes with dual fog"**
**Solution:** Check batch unpacking logic - ensure dataloader returns 4 items in dual fog mode.

---

## 🔍 Code Structure

```
Model Forward Pass (Training):
├── Input: (3B, 3, 640, 640)  # view1 + view2 + clean
├── Backbone
│   ├── feat1, feat2, feat3
│   └── dehazing output
├── SeverityHead(feat3) → s(x) → λ(x)
├── Neck (FPN)
│   ├── P3, P4, P5
│   └── Clone for supervision
├── SpatialWeightHeads(P3,P4,P5) → W3,W4,W5
├── YOLO Heads → detections
└── Return: {detections, dehazing, neck_features, severity, weights}

Training Step:
├── Split features: view1 / view2 / clean
├── L_det = YOLO loss on view1
├── L_dehazy = MSE(dehaze, clean)
├── L_align = WeightedMSE(features_v1, features_clean, W)
├── L_con = InfoNCE(features_v1, features_v2)
└── L_total = L_det + λ(x)*L_align + β*L_con
```

---

## 📈 Hyperparameter Tuning Guide

### **Initial Values (Recommended)**
- λ_min = 0.05, λ_max = 0.20 (keep alignment moderate)
- β = 0.1 (start with 10% of detection loss)
- τ = 0.2 (standard for contrastive learning)

### **If Detection Performance Drops**
- Decrease λ_max (reduce feature alignment strength)
- Decrease β (reduce contrastive weight)
- Increase warmup epochs

### **If Fog Robustness Not Improving**
- Increase β (strengthen invariance)
- Increase fog intensity range in dataloader
- Check λ(x) values in logs (should vary 0.05-0.20)

### **If Training Unstable**
- Reduce learning rate
- Increase warmup epochs (train task heads longer)
- Check W normalization (weights shouldn't collapse to 0)

---

## ✅ Validation

### **Check Task Heads Are Working**
```python
# During training, verify in logs:
- λ_avg should vary between λ_min and λ_max
- align loss should be > 0
- contr loss should be > 0
```

### **Verify Dual Fog Views**
```python
# In dataloader __getitem__:
assert len(batch[0]) == 4  # view1, view2, labels, clean
```

### **Test Forward Pass**
```python
model = YoloBody(anchors_mask, num_classes, use_task_heads=True)
x = torch.randn(6, 3, 640, 640)  # 2 images * 3 (v1+v2+clean)
outputs = model(x)
assert 'severity' in outputs
assert len(outputs['neck_features']) == 3  # P3, P4, P5
```

---

## 📚 References

Implementation based on:
- Task-guided contrastive learning principles
- Adaptive loss weighting (severity-based)
- Spatial attention for feature alignment
- InfoNCE contrastive objective

---

## 🎓 Citation

If you use this implementation, cite both RDFNet and your task-guided approach.

---

**Status:** ✅ Implementation complete and tested
**Compatibility:** Backward compatible with original RDFNet
**Requirements:** PyTorch >= 1.9, CUDA-enabled GPU
