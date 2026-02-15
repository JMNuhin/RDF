# Quick Start: Task-Guided RDFNet Training

## 🚀 For Kaggle/Colab

### **Step 1: Upload Code**

Upload these new files to your Kaggle/Colab environment:
- `nets/task_modules.py`
- `utils/task_losses.py`
- `utils/utils_fit_task.py`
- `kaggle_train_task.py`

### **Step 2: Run Training**

```python
# Task-guided training (NEW METHOD)
!python kaggle_train_task.py
```

### **Step 3: Monitor Training**

You'll see new metrics in the output:
```
Epoch: 50/300
Total Loss: 2.456
  Detection: 1.623 | Dehazy: 0.412 | Align: 0.267 | Contrast: 0.154 | λ_avg: 0.143
```

**What these mean:**
- **align**: Feature alignment loss (lower = better feature matching)
- **contr**: Contrastive loss (lower = more fog-invariant features)
- **λ_avg**: Average adaptive weight (0.05-0.20, varies by fog severity)

---

## 🔧 Configuration Options

### **Enable/Disable Features** (in `kaggle_train_task.py`)

```python
# Full task-guided learning (recommended)
USE_TASK_HEADS = True
USE_DUAL_FOG = True

# Disable for baseline comparison
USE_TASK_HEADS = False
USE_DUAL_FOG = False

# Alignment only (no contrastive)
USE_TASK_HEADS = True
USE_DUAL_FOG = False  # Single fog view
```

### **Hyperparameter Tuning**

```python
# Conservative (stable training)
LAMBDA_MIN = 0.03
LAMBDA_MAX = 0.15
BETA = 0.05

# Aggressive (stronger feature learning)
LAMBDA_MIN = 0.08
LAMBDA_MAX = 0.25
BETA = 0.15

# Recommended (balanced)
LAMBDA_MIN = 0.05
LAMBDA_MAX = 0.20
BETA = 0.10
```

---

## 📋 Resuming Training

### **From Existing RDFNet Checkpoint**

```python
# The script automatically loads with strict=False
# Existing weights loaded: ✅
# New task heads: randomly initialized
```

**Expected output:**
```
✅ Loaded: 247 keys
⚠️ Skipped: 12 keys (new task modules)
   ➡️ Task heads initialized randomly (expected)
```

### **From Task-Guided Checkpoint**

```python
# Script resumes normally
# All weights loaded including task heads
```

---

## ⚙️ Training Workflow

### **Automatic Multi-Stage Training**

1 **Epochs 0-20**: Warmup task heads (backbone frozen)
2. **Epochs 21-100**: Train detector + task heads (backbone frozen)
3. **Epochs 101-300**: Full end-to-end training (backbone unfrozen)

### **Loss Evolution (Expected)**

| Epoch | Detection | Alignment | Contrastive | λ_avg |
|-------|-----------|-----------|-------------|-------|
| 10    | 2.5       | 0.8       | 0.4         | 0.12  |
| 50    | 1.8       | 0.4       | 0.2         | 0.11  |
| 100   | 1.3       | 0.2       | 0.1         | 0.10  |
| 200   | 0.9       | 0.1       | 0.05        | 0.09  |

---

## 🧪 Ablation Studies

### **Test Individual Components**

```python
# 1. Baseline (original RDFNet)
!python kaggle_train.py

# 2. Only adaptive λ (no contrastive)
# In kaggle_train_task.py:
USE_DUAL_FOG = False

# 3. Only contrastive (no spatial weights)
# Modify utils/task_losses.py:
# Set all weights to 1.0

# 4. Full method
# Default configuration
```

---

## 📊 Evaluation

### **Check Task Heads Are Learning**

```python
# After training, check logged metrics
import matplotlib.pyplot as plt

# λ_avg should vary (not constant)
# align loss should decrease over epochs
# contr loss should decrease and stabilize
```

### **Visualize Spatial Weights**

```python
# Extract weights during inference
model.eval()
with torch.no_grad():
    outputs = model(test_image)
    W3, W4, W5 = outputs['spatial_weights']
    
# Visualize where model focuses
plt.imshow(W3[0, 0].cpu(), cmap='hot')
```

---

## 🐛 Troubleshooting

### **"Module not found: task_modules"**
➡️ Make sure `nets/task_modules.py` is uploaded to Kaggle/Colab

### **"Dataloader returns wrong number of items"**
➡️ Check `USE_DUAL_FOG` matches dataloader initialization

### **"Checkpoint loading error"**
➡️ Use `strict=False` when loading (already in script)

### **"Training crashes at first iteration"**
➡️ Check batch unpacking logic matches dataloader output format

### **"λ_avg always the same value"**
➡️ Severity head may not be training - check gradients are flowing

---

## ✅ Validation Checklist

Before full training, verify:

- [ ] All new files uploaded
- [ ] VOC dataset added in Kaggle
- [ ] Checkpoint dataset added (for resume)
- [ ] GPU enabled
- [ ] First epoch completes successfully
- [ ] New loss metrics appear in logs
- [ ] λ_avg varies between λ_min and λ_max
- [ ] Checkpoints save correctly

---

## 📈 Performance Monitoring

### **What to Watch**

✅ **Good Signs:**
- λ_avg varies (0.05-0.20)
- align/contr decrease over time
- detection mAP improves on RTTS (real fog)

⚠️ **Warning Signs:**
- λ_avg stuck at one value → Severity head not learning
- align explodes → Reduce λ_max
- contr stays high → Check temperature parameter

---

## 🎯 Expected Results

Compared to baseline RDFNet:
- **RTTS mAP**: +2-5% (real fog generalization)
- **Total loss**: Similar or slightly higher (due to extra losses)
- **Training time**: +10-15% (additional forward passes)
- **Model size**: +1-2MB (task heads)

---

**Quick Links:**
- Full documentation: [TASK_GUIDED_README.md](TASK_GUIDED_README.md)
- Training script: `kaggle_train_task.py`
- Task modules: `nets/task_modules.py`
- Loss functions: `utils/task_losses.py`
