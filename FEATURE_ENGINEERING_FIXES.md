# Feature Engineering Fixes for OS Fingerprinting

## Problem Summary

The original Model 2b training showed concerning feature importance scores that indicate **spurious correlations** rather than legitimate OS fingerprinting:

```
Feature Importance (ORIGINAL):
1. ip_id           : 24.3%  ❌ TOO HIGH
2. dst_port        : 22.9%  ❌ APPLICATION-SPECIFIC
3. src_port        : 15.9%  ❌ APPLICATION-SPECIFIC
4. tcp_timestamp   : 11.7%  ✓ Legitimate
5. tcp_window_size :  6.7%  ✓ Legitimate (but should be higher!)
```

**Total port importance: 38.7%** - This is a red flag!

---

## Root Causes

### 🚨 Issue 1: Port Numbers Are Application-Specific

**Problem:**
- Port numbers (80, 443, 22, etc.) identify **applications**, not operating systems
- If your dataset has Windows machines using HTTPS (port 443) more often, the model learns:
  - **"port 443 → Windows"** (WRONG!)
  - Instead of **"TCP window size 64240 → Windows"** (CORRECT!)

**Why this happens:**
- Data collection bias: Different OSes might run different applications
- Temporal patterns: Windows machines collected during web browsing sessions
- Network topology: Certain subnets have specific application profiles

**Impact:**
- Model won't generalize to new deployments
- Accuracy will drop dramatically in production
- Model is learning "what apps run on what OS" not "OS fingerprints"

---

### 🚨 Issue 2: IP ID Dominance (24.3% importance)

**Problem:**
While IP ID **does** have OS-specific behavior (Windows: sequential, Linux: random), it should NOT be the top feature.

**Why IP ID is problematic in this dataset:**

1. **Single-packet limitation:**
   - CESNET preprocessing extracts **only SYN packets** (`syn_only=True`)
   - You typically get **ONE packet per connection**
   - Cannot determine sequential/random behavior from a single packet!
   - Need multiple packets from the same source IP to analyze increments

2. **Overfitting risk:**
   - Raw IP ID values can encode:
     - Temporal patterns (when data was collected)
     - Network-specific behavior
     - Not just OS-specific patterns

**Evidence of overfitting:**
- IP ID being more important than TCP window size and TCP options is backwards
- True OS fingerprinting relies on TCP handshake parameters, not IP header artifacts

---

## Solutions Implemented

### ✅ Fix 1: Remove Port Numbers

**Changed:**
```python
# BEFORE (train_model2b_modern.py)
network_features = [
    'src_port',      # ❌ Application-specific
    'dst_port',      # ❌ Application-specific
    'protocol',
]

# AFTER (train_model2b_modern_fixed.py)
network_features = [
    'protocol',      # ✓ Legitimate (TCP vs UDP)
]
```

**Added port categories instead:**
```python
# Behavioral categories (OS might prefer certain port ranges)
port_features = [
    'src_port_is_well_known',    # < 1024
    'src_port_is_ephemeral',     # >= 49152
    'dst_port_is_well_known',
    'dst_port_is_ephemeral',
    'dst_port_is_http',          # Common services
    'dst_port_is_https',
    'dst_port_is_ssh',
    ...
]
```

This captures **behavioral patterns** (e.g., "Windows prefers higher ephemeral ports") without overfitting to specific port numbers.

---

### ✅ Fix 2: Remove IP ID (for single-packet datasets)

**Decision:** Remove IP ID entirely for CESNET dataset

**Rationale:**
- Cannot extract behavioral features from single SYN packets
- Would need multiple packets per host to analyze:
  - Increment rate (Windows: ~1 per packet, Linux: random)
  - Variance (sequential vs random distribution)
  - Temporal patterns

**If you want to use IP ID in the future:**

1. **Reprocess dataset with all TCP packets:**
   ```bash
   python preprocess_scripts/cesnet_preprocess.py --all-tcp
   ```

2. **Group by source IP and extract behavioral features:**
   ```python
   def extract_ip_id_sequential_behavior(df):
       """Requires multiple packets per source IP"""
       grouped = df.groupby('src_ip')['ip_id'].agg([
           ('ip_id_increment_mean', lambda x: x.diff().mean()),
           ('ip_id_increment_std', lambda x: x.diff().std()),
           ('ip_id_appears_sequential', lambda x: x.diff().std() < 100)
       ])
       return grouped
   ```

3. **Use behavioral features, not raw values**

---

### ✅ Fix 3: Remove TTL (redundant with initial_ttl)

**Problem: Perfect Multicollinearity**

The preprocessing extracts both `ttl` and `initial_ttl`:

```python
# cesnet_preprocess.py:254-255
'ttl': ip_layer.ttl,                      # Observed TTL (e.g., 61, 125, 253)
'initial_ttl': calculate_initial_ttl(ttl), # Estimated initial (e.g., 64, 128, 255)
```

Since `initial_ttl` is **deterministically calculated** from `ttl`, they are perfectly correlated!

**Why keep only initial_ttl?**

1. **OS fingerprint:** Initial TTL is what the OS sets (64=Linux, 128=Windows, 255=Cisco)
2. **Network-independent:** Same value regardless of how many router hops
3. **Industry standard:** p0f, Nmap, and PRADS all use initial TTL for OS detection
4. **No information loss:** `ttl` provides no additional signal

**Feature importance evidence:**
```
Original model:
  ttl:         2.0% importance  ─┐
  initial_ttl: 2.4% importance  ─┤ Combined: 4.4% (split across redundant features)
                                 ─┘
Fixed model:
  initial_ttl: ~4-5% importance  (all signal in one feature)
```

---

### ✅ Fix 4: Increased Model Regularization

**Prevent single features from dominating:**

```python
# BEFORE
params = {
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
}

# AFTER
params = {
    'max_depth': 12,              # ↓ Reduced depth
    'min_samples_split': 10,      # ↑ More samples required
    'min_samples_leaf': 5,        # ↑ Larger leaves
    'max_features': 'log2',       # ↓ Fewer features per split
    'max_leaf_nodes': 100,        # NEW: Limit tree complexity
}
```

This forces the model to learn from **multiple features** instead of relying on one dominant feature.

---

## Expected Results After Fixes

### Feature Importance (EXPECTED with fixes):

```
Top Features (should be):
1. tcp_window_size      : ~15-20%  ✓ Core OS fingerprint
2. tcp_options_order    : ~15-20%  ✓ Core OS fingerprint
3. tcp_window_scale     : ~10-15%  ✓ Core OS fingerprint
4. tcp_mss              : ~8-12%   ✓ Core OS fingerprint
5. initial_ttl          : ~8-10%   ✓ Core OS fingerprint
6. tcp_timestamp_val    : ~5-8%    ✓ Core OS fingerprint
```

**These are LEGITIMATE OS fingerprinting features used in:**
- [p0f](https://lcamtuf.coredump.cx/p0f3/) (passive OS fingerprinting)
- [Nmap](https://nmap.org/book/osdetect.html) (active OS detection)
- [PRADS](https://github.com/gamelinux/prads) (passive network detection)

---

## How to Use the Fixed Version

### Option 1: Quick Test (recommended)
```bash
# Use the fixed training script
python train_scripts/train_model2b_modern_fixed.py \
    --input datasets/cesnet_merged.csv \
    --merge-classes \
    --cross-validate
```

### Option 2: Compare with Original
```bash
# Train original (with port/IP ID issues)
python train_scripts/train_model2b_modern.py \
    --input datasets/cesnet_merged.csv \
    --merge-classes

# Train fixed version
python train_scripts/train_model2b_modern_fixed.py \
    --input datasets/cesnet_merged.csv \
    --merge-classes

# Compare feature importance in results/
diff results/model2b_evaluation.json \
     results/model2b_evaluation_fixed.json
```

---

## Validation Checklist

After training with the fixed version, verify:

- [ ] **No port numbers** in top 10 features
- [ ] **No IP ID** in top 10 features
- [ ] **TCP window size** in top 3
- [ ] **TCP options order** in top 3
- [ ] **TTL or initial_ttl** in top 10
- [ ] **TCP MSS** in top 10

If these conditions are met, your model is learning **legitimate OS fingerprints**!

---

## Further Improvements

### 1. Collect More Data
- Current dataset: ~1,800 samples across 13 OS versions
- Recommended: 500+ samples per OS version for production
- Use data augmentation (SMOTE/ADASYN) carefully

### 2. Feature Engineering
- **TCP timestamp granularity:** Extract timestamp increment rate
  - Linux: ~1ms (HZ=1000)
  - Windows: ~100ms
  - macOS: ~10ms

- **TCP options interactions:** Combine options
  - Example: "MSS=1460 AND WScale=8" is Windows 10/11 signature

### 3. Multi-Packet Analysis
- Extract features from multiple packets per flow
- Calculate variance in TCP parameters
- Detect retransmission behavior (OS-specific)

---

## References

1. **p0f v3 Documentation:** https://lcamtuf.coredump.cx/p0f3/README
   - Industry-standard passive OS fingerprinting
   - Lists legitimate TCP/IP fingerprinting features

2. **RFC 1323 - TCP Extensions:** https://tools.ietf.org/html/rfc1323
   - Window scaling and timestamps (OS-specific implementations)

3. **Nmap OS Detection:** https://nmap.org/book/osdetect.html
   - Active OS fingerprinting techniques

4. **Research Paper:** "SoK: Exploring the State of the Art and the Future of Operating System Fingerprinting"
   - Comprehensive survey of OS fingerprinting methods

---

## Summary

| Issue | Impact | Fix |
|-------|--------|-----|
| Port numbers in features | Learning app patterns, not OS | Removed raw ports, added categories |
| IP ID too important | Overfitting to temporal patterns | Removed (single-packet limitation) |
| TTL and initial_ttl redundant | Multicollinearity, diluted importance | Keep only initial_ttl (OS fingerprint) |
| Single feature dominance | Brittle model, poor generalization | Increased regularization |

**Bottom line:** The original model was learning **"what apps run on Windows"** instead of **"what makes Windows TCP stack unique"**. The fixed version focuses on legitimate OS fingerprints.
