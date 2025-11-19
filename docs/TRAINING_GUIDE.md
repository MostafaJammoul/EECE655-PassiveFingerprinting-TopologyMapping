# Model Training Guide

This guide covers training all three models for the hierarchical OS fingerprinting system.

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Model 1: OS Family Classifier                         │
│  Dataset: Masaryk (flow-level features)                │
│  Algorithm: XGBoost                                     │
│  Classes: Windows, Linux, macOS, Android, iOS, BSD     │
│  Expected Accuracy: 85-95%                              │
└─────────────────────────────────────────────────────────┘
                          ↓
         ┌────────────────┴────────────────┐
         ↓                                  ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│  Model 2a: Legacy OS     │  │  Model 2b: Modern OS     │
│  Dataset: nPrint         │  │  Dataset: CESNET Idle    │
│  Algorithm: XGBoost      │  │  Algorithm: Random Forest│
│  ~48k samples            │  │  ~1.8k samples           │
│  Expected: 85-90%        │  │  Expected: 70-85%        │
└──────────────────────────┘  └──────────────────────────┘
```

## 🚀 Quick Start

### Install Dependencies

```bash
pip install xgboost scikit-learn imbalanced-learn matplotlib seaborn pandas numpy
```

### Train All Models (Recommended Settings)

```bash
# Train Model 1 (OS Family)
python scripts/train_model1_family.py

# Train Model 2a (Legacy OS)
python scripts/train_model2a_legacy.py --use-smote

# Train Model 2b (Modern OS) - IMPORTANT: Use recommended flags!
python scripts/train_model2b_modern.py \
    --merge-classes \
    --use-adasyn \
    --cross-validate
```

## 📋 Detailed Training Instructions

### Model 1: OS Family Classifier

**Dataset:** Masaryk (flow-level TCP SYN features)
**Size:** Medium-large dataset
**Algorithm:** XGBoost (best for this size)

#### Basic Training

```bash
python scripts/train_model1_family.py \
    --input data/processed/masaryk_processed.csv \
    --output-dir models \
    --results-dir results
```

#### Options

- `--test-size`: Fraction for testing (default: 0.2)
- `--val-size`: Fraction for validation (default: 0.1)
- `--random-state`: Random seed (default: 42)
- `--quiet`: Suppress output

#### Expected Performance

- **Accuracy:** 85-95%
- **Training time:** 2-5 minutes
- **Features:** TCP/IP fingerprints, flow statistics

#### Key Features for Model 1

- TCP window size & scale
- Initial TTL
- TCP MSS (Maximum Segment Size)
- TCP options (SACK, Timestamp, NOP)
- Flow duration and packet rates

---

### Model 2a: Legacy OS Classifier

**Dataset:** nPrint (packet-level TCP SYN features)
**Size:** ~48,000 samples
**Algorithm:** XGBoost
**Target OSs:** Windows 7/8, Ubuntu 14.04/16.04, CentOS, Debian 8/9

#### Recommended Training

```bash
python scripts/train_model2a_legacy.py \
    --input data/processed/nprint_packets.csv \
    --use-smote \
    --smote-threshold 100
```

#### Options

- `--use-smote`: Apply SMOTE for minority classes (RECOMMENDED)
- `--smote-threshold`: Min samples before SMOTE (default: 100)
- `--test-size`: Test fraction (default: 0.2)
- `--val-size`: Validation fraction (default: 0.1)

#### When to Use SMOTE

✅ **Use SMOTE if:**
- Any OS class has < 100 samples
- You see poor performance on minority classes
- Class distribution is heavily imbalanced (>10:1 ratio)

❌ **Skip SMOTE if:**
- All classes have > 200 samples
- Dataset is naturally balanced

#### Expected Performance

- **Accuracy:** 85-90%
- **Training time:** 3-7 minutes
- **Challenging classes:** CentOS (often has fewer samples)

#### Key Features for Model 2a

- **TCP options order** (HIGHLY discriminative!)
  - Example: "MSS:NOP:WS:NOP:NOP:SACK" uniquely identifies certain OSs
- TCP window size (varies by OS version)
- Initial TTL (128 for Windows, 64 for Linux)
- TCP MSS values
- Don't Fragment (DF) flag

---

### Model 2b: Modern OS Classifier

**Dataset:** CESNET Idle (packet-level TCP SYN features)
**Size:** ~1,800 samples ⚠️ **SMALL DATASET**
**Algorithm:** Random Forest (more robust for small data)
**Target OSs:** Win10/11, Ubuntu 22/24, Fedora 36+, macOS 13+, Android

#### ⚠️ CRITICAL: Recommended Settings

Due to the small dataset size, **ALWAYS use these flags:**

```bash
python scripts/train_model2b_modern.py \
    --merge-classes \
    --use-adasyn \
    --cross-validate \
    --adasyn-threshold 50
```

#### Why These Settings?

**`--merge-classes`** (HIGHLY RECOMMENDED)
- Combines similar OS versions to increase samples per class
- Examples:
  - Windows 10 + Windows 11 → "Windows 10/11"
  - Ubuntu 22.04 + 24.04 → "Ubuntu 22+"
  - macOS 13 + 14 + 15 → "macOS 13+"
- **Impact:** Increases samples per class from ~120 to ~300+

**`--use-adasyn`** (HIGHLY RECOMMENDED)
- ADASYN (Adaptive Synthetic Sampling) generates synthetic samples
- Better than SMOTE for very imbalanced data
- Brings minority classes up to threshold

**`--cross-validate`** (RECOMMENDED)
- Performs 5-fold cross-validation
- Gives more reliable performance estimate
- Critical for small datasets

#### Expected Performance

**Without recommended flags:**
- Accuracy: 50-70% (likely overfitting)
- High variance across runs
- Poor generalization

**With recommended flags:**
- Accuracy: 70-85%
- More stable performance
- Better generalization

#### Options

- `--merge-classes`: Merge similar OS versions (RECOMMENDED)
- `--use-adasyn`: Apply ADASYN balancing (RECOMMENDED)
- `--cross-validate`: Perform k-fold CV (RECOMMENDED)
- `--adasyn-threshold`: Min samples per class (default: 50)
- `--test-size`: Test fraction (default: 0.2)

#### Training Time

- Without ADASYN: 1-2 minutes
- With ADASYN: 2-4 minutes
- With cross-validation: 5-10 minutes

---

## 🎯 Algorithm Choices: Why These?

### Model 1: XGBoost ✅

**Advantages:**
- Handles medium-large datasets excellently
- Built-in missing value handling
- Robust to class imbalance with `scale_pos_weight`
- Generally 2-5% better accuracy than Random Forest
- Fast training with early stopping

**Why not Random Forest?**
- XGBoost typically outperforms RF on flow-level features
- Gradient boosting captures more complex patterns

**Why not SVM?**
- Doesn't scale well to large datasets
- Requires extensive feature scaling
- Slower training

### Model 2a: XGBoost ✅

**Advantages:**
- ~48k samples is ideal for XGBoost
- Packet-level features benefit from boosting
- Handles class imbalance well

**Why not Random Forest?**
- XGBoost typically 3-7% better on this task
- Better handles TCP options encoding

### Model 2b: Random Forest ✅

**Advantages:**
- **More robust to overfitting with small data (~1.8k samples)**
- Less sensitive to hyperparameter tuning
- Natural ensemble averaging reduces variance
- Works well with `class_weight` for imbalanced data

**Why not XGBoost?**
- ⚠️ **XGBoost tends to overfit badly on <2k samples**
- Requires extensive hyperparameter tuning
- More prone to memorizing training data

**Why not SVM?**
- Doesn't handle class imbalance as well
- Slower training

---

## 📊 Sampling Strategies

### Stratified Sampling (All Models) ✅

**Always used** in train/test splits to preserve class distribution.

```python
train_test_split(X, y, stratify=y)  # Preserves class ratios
```

### SMOTE (Model 2a - Optional) ⚠️

**Use when:** Minority classes have < 100 samples

**How it works:**
1. Identifies minority classes
2. Creates synthetic samples using k-nearest neighbors
3. Brings classes up to threshold

**Example:**
```bash
# Apply SMOTE to classes with < 100 samples
python scripts/train_model2a_legacy.py --use-smote --smote-threshold 100
```

**Risks:**
- Can create unrealistic samples if used too aggressively
- May not help if class has < 10 samples

### ADASYN (Model 2b - Recommended) ✅

**Use for Model 2b:** Handles extreme class imbalance better than SMOTE

**How it works:**
1. Adaptive: Creates more synthetic samples for harder-to-learn minority samples
2. Uses k-nearest neighbors adaptively
3. Falls back to SMOTE if ADASYN fails

**Example:**
```bash
# Apply ADASYN to classes with < 50 samples
python scripts/train_model2b_modern.py --use-adasyn --adasyn-threshold 50
```

### Class Weighting (All Models) ✅

**Always used** to handle imbalanced classes.

```python
# Automatically calculated
class_weight = {cls: max_count / count for cls, count in class_distribution}
```

---

## 🔍 Evaluation Metrics

All training scripts generate:

### 1. Overall Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: TP / (TP + FP) - weighted average
- **Recall**: TP / (TP + FN) - weighted average
- **F1-Score**: Harmonic mean of precision & recall

### 2. Per-Class Metrics
- Individual precision/recall/F1 for each OS
- Support (number of test samples)

### 3. Confusion Matrix
- Visualizes misclassifications
- Saved as PNG in `results/`

### 4. Feature Importance
- Top 20 most important features
- Saved as PNG in `results/`

### 5. Results JSON
- Complete metrics in machine-readable format
- Saved in `results/modelX_evaluation.json`

---

## 📁 Output Files

After training, you'll have:

```
models/
├── model1_os_family.pkl           # Trained Model 1
├── model1_feature_names.pkl       # Feature list
├── model1_label_encoder.pkl       # Class mappings
├── model1_class_weights.pkl       # Class weights
├── model2a_legacy_os.pkl          # Trained Model 2a
├── model2a_feature_names.pkl
├── model2a_label_encoder.pkl
├── model2a_class_weights.pkl
├── model2b_modern_os.pkl          # Trained Model 2b
├── model2b_feature_names.pkl
├── model2b_label_encoder.pkl
└── model2b_class_weights.pkl

results/
├── model1_evaluation.json         # Metrics
├── model1_confusion_matrix.png
├── model1_feature_importance.png
├── model2a_evaluation.json
├── model2a_confusion_matrix.png
├── model2a_feature_importance.png
├── model2b_evaluation.json
├── model2b_confusion_matrix.png
└── model2b_feature_importance.png
```

---

## 🚨 Common Issues & Solutions

### Issue: Model 2b has low accuracy (< 60%)

**Solutions:**
1. ✅ Use `--merge-classes` to combine similar OS versions
2. ✅ Use `--use-adasyn` to balance classes
3. ✅ Check if some classes have < 20 samples (may need to drop them)
4. Run `--cross-validate` to get better performance estimate

### Issue: SMOTE/ADASYN fails with error

**Cause:** Some class has too few samples for k-neighbors

**Solutions:**
1. Reduce k-neighbors (automatic fallback in scripts)
2. Remove classes with < 5 samples
3. Merge very small classes into "Other"

### Issue: Training takes too long

**Solutions:**
1. Reduce `n_estimators` in model parameters
2. Use `--quiet` flag to suppress output
3. Reduce validation set size with `--val-size 0.05`

### Issue: Out of memory

**Solutions:**
1. Reduce `n_estimators` (e.g., from 300 to 200)
2. Use `--quiet` to reduce memory overhead
3. Process datasets in chunks (advanced)

---

## 💡 Best Practices

### 1. Always Preprocess First
```bash
python scripts/preprocess_masaryk.py
python scripts/preprocess_nprint.py
python scripts/preprocess_cesnet_idle.py
```

### 2. Train in Order
Train Model 1 first to understand OS family distribution, which informs Model 2a/2b training.

### 3. Monitor Per-Class Performance
Some OSs may have inherently low accuracy due to similar TCP stacks (e.g., Ubuntu variants).

### 4. Use Cross-Validation for Model 2b
With only ~1.8k samples, a single train/test split may not be representative.

### 5. Save Training Logs
```bash
python scripts/train_model2b_modern.py --merge-classes --use-adasyn 2>&1 | tee training_log.txt
```

### 6. Compare Configurations
Try different settings and compare `results/model*_evaluation.json`:

```bash
# Without merging
python scripts/train_model2b_modern.py --use-adasyn

# With merging
python scripts/train_model2b_modern.py --merge-classes --use-adasyn

# Compare accuracy in JSON files
```

---

## 📈 Expected Results Summary

| Model | Dataset | Algorithm | Expected Accuracy | Training Time |
|-------|---------|-----------|-------------------|---------------|
| 1 (Family) | Masaryk | XGBoost | 85-95% | 2-5 min |
| 2a (Legacy) | nPrint | XGBoost | 85-90% | 3-7 min |
| 2b (Modern) | CESNET | Random Forest | 70-85%* | 2-10 min |

*With `--merge-classes` and `--use-adasyn`

---

## 🔄 Next Steps After Training

1. **Evaluate Results**: Check confusion matrices for systematic errors
2. **Build Ensemble**: Create routing logic to combine models
3. **Test on New Data**: Validate on unseen PCAP files
4. **Iterate**: Adjust class merging or feature selection based on results

See `scripts/predict_ensemble.py` for inference with all three models.
