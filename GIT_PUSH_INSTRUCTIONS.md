# 📤 GitHub Push Instructions

## Step-by-Step Guide to Push RDFNet to GitHub

### **Step 1: Initialize Git Repository**

Open PowerShell in the project directory and run:

```powershell
cd C:\Users\User\Desktop\rdf\RDF_net-main

# Initialize git repository
git init

# Check status
git status
```

---

### **Step 2: Create GitHub Repository**

1. Go to [GitHub](https://github.com)
2. Click **"New repository"** (+ icon in top right)
3. Repository name: `RDFNet-Task-Guided` (or your preferred name)
4. Description: "RDFNet with Task-Guided Contrastive Learning for Foggy Object Detection"
5. **Public** or **Private** (your choice)
6. **DO NOT** initialize with README (we already have one)
7. Click **"Create repository"**

GitHub will show you commands - **keep that page open**.

---

### **Step 3: Configure Git (First Time Only)**

If you haven't configured Git before:

```powershell
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify
git config --global --list
```

---

### **Step 4: Add Files to Git**

```powershell
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status

# You should see:
# - New files in green (all .py, .md, .txt files)
# - Ignored files won't show (*.pth, __pycache__, etc.)
```

---

### **Step 5: Create First Commit**

```powershell
git commit -m "Initial commit: RDFNet with task-guided contrastive learning

- Implemented severity-adaptive loss weighting
- Added spatially adaptive feature supervision
- Integrated InfoNCE contrastive learning
- Dual fog view generation for training
- Complete documentation and guides"
```

---

### **Step 6: Link to GitHub Repository**

Replace `YOUR_USERNAME` and `REPO_NAME` with your actual GitHub info:

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Verify remote
git remote -v
```

**Example:**
```powershell
git remote add origin https://github.com/johndoe/RDFNet-Task-Guided.git
```

---

### **Step 7: Push to GitHub**

```powershell
# Push main branch
git branch -M main
git push -u origin main
```

**If prompted for credentials:**
- **Username:** Your GitHub username
- **Password:** Use a **Personal Access Token** (not your GitHub password)

**To create a token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Check "repo" scope → Generate
3. Copy token (save it - you won't see it again!)
4. Use token as password when pushing

---

### **Step 8: Verify Upload**

Go to your GitHub repository URL:
```
https://github.com/YOUR_USERNAME/REPO_NAME
```

You should see:
- ✅ All Python files (`.py`)
- ✅ Documentation files (`.md`)
- ✅ Configuration files (`.txt`)
- ✅ Folder structure (`nets/`, `utils/`, etc.)
- ❌ No large `.pth` files
- ❌ No `__pycache__` folders

---

## 🔄 Future Updates

When you make changes locally:

```powershell
# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## 🚨 Troubleshooting

### **"Large files detected"**

If Git complains about large files (e.g., `.pth` weights):

```powershell
# Check .gitignore includes them
cat .gitignore | Select-String "pth"

# Remove from staging if accidentally added
git rm --cached model_data/RDFNet.pth
git commit -m "Remove large model file"
```

### **"Repository not found"**

Check remote URL:
```powershell
git remote -v
# Fix if wrong:
git remote set-url origin https://github.com/CORRECT_USERNAME/CORRECT_REPO.git
```

### **"Authentication failed"**

Use a Personal Access Token instead of password:
1. Generate token on GitHub (see Step 7)
2. Use token when prompted for password

**Or cache credentials:**
```powershell
git config --global credential.helper wincred
```

### **"Permission denied"**

If using SSH instead of HTTPS:
```powershell
# Switch to HTTPS
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

---

## 📋 Quick Commands Reference

```powershell
# Initialize repo
git init

# Add all files
git add .

# Commit
git commit -m "Your message"

# Add remote
git remote add origin https://github.com/USER/REPO.git

# Push
git push -u origin main

# Check status
git status

# View history
git log --oneline

# View remotes
git remote -v
```

---

## 🎯 Next Step: Use in Kaggle

Once pushed to GitHub, follow [KAGGLE_SETUP.md](KAGGLE_SETUP.md) to:
1. Clone in Kaggle notebook
2. Add datasets
3. Start training

---

## ✅ Checklist

- [ ] Git initialized (`git init`)
- [ ] Files added (`git add .`)
- [ ] First commit created
- [ ] GitHub repository created
- [ ] Remote added (`git remote add origin ...`)
- [ ] Code pushed (`git push -u origin main`)
- [ ] Verified on GitHub web interface
- [ ] Ready to clone in Kaggle!

---

**Your repository is now public/shareable and ready for Kaggle! 🎉**
