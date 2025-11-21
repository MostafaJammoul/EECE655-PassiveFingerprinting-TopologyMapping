# Feature Extraction Review and Enhancement Plan

## Current Feature Extraction Status

### Dataset Overview

1. **Masaryk Dataset** - ✅ FLOW-BASED
   - Format: CSV with semicolon-separated flow-level statistics
   - Target: OS Family (Windows, Linux, macOS, Android, iOS)
   - Filtering: TCP flows with SYN flag present only

2. **nprint Dataset** - PACKET-BASED
   - Format: PCAPNG with packet comments containing OS labels
   - Target: Specific OS Version
   - Filtering: Configurable (SYN-only or all TCP)

3. **CESNET Idle Dataset** - PACKET-BASED (99.9% ACK packets)
   - Format: PCAP files organized by OS subdirectories
   - Target: Specific OS Version
   - Filtering: Configurable (SYN-only or all TCP)

---

## Currently Extracted Features

### ✅ Already Implemented

#### IP Layer Features
- ✅ `ip_tos` - Type of Service (MEDIUM importance)
- ✅ `ttl` - Time To Live
- ✅ `initial_ttl` - Estimated original TTL (64/128/255)
- ✅ `df_flag` - Don't Fragment flag (HIGH importance)
- ✅ `ip_id` - IP Identification (CRITICAL - Windows incremental, Linux random)
- ✅ `ip_len` - IP packet length

#### TCP Layer Features
- ✅ `tcp_window_size` - TCP window size
- ✅ `tcp_flags` - TCP flags bitmap
- ✅ `tcp_mss` - Maximum Segment Size (HIGH importance)
- ✅ `tcp_window_scale` - Window Scale factor (HIGH importance)
- ✅ `tcp_sack_permitted` - SACK permitted flag (MEDIUM importance)
- ✅ `tcp_timestamp_val` - TCP timestamp value (CRITICAL)
- ✅ `tcp_timestamp_ecr` - TCP timestamp echo reply (CRITICAL)
- ✅ `tcp_options_order` - Order of TCP options (HIGHLY discriminative!)
- ✅ `tcp_urgent_ptr` - Urgent pointer (LOW importance)

#### Flow-Level Features (Masaryk only)
- ✅ `flow_duration` - Duration of flow
- ✅ `pkt_count_forward` / `pkt_count_backward` - Bidirectional packet counts
- ✅ `bytes_sent` / `bytes_received` - Bidirectional byte counts
- ✅ `max_ttl_forward` / `max_ttl_backward` - Bidirectional max TTL

#### Metadata
- ✅ `timestamp` - Packet/flow timestamp
- ✅ `src_ip` / `dst_ip` - IP addresses
- ✅ `src_port` / `dst_port` - Port numbers
- ✅ `protocol` - Protocol number

---

## ❌ Missing Features (Requested)

### 1. Sequence Number (tcp.seq)
- **Status**: NOT extracted
- **Importance**: HIGH for ACK-based fingerprinting
- **Use case**:
  - Initial Sequence Number (ISN) reveals OS randomization patterns
  - Sequence number increments can distinguish OS TCP stacks
- **Where to add**:
  - `nprint_preprocess.py` (line ~380)
  - `cesnet_preprocess.py` (line ~265)

### 2. Acknowledgment Number (tcp.ack)
- **Status**: NOT extracted
- **Importance**: MEDIUM for ACK-based fingerprinting
- **Use case**:
  - ACK behavior differs across OSes
  - Delayed ACK timers vary by OS
- **Where to add**: Same as sequence number

### 3. Inter-Packet Timing (IPT)
- **Status**: NOT extracted
- **Importance**: HIGH for behavioral fingerprinting
- **Use case**:
  - Different OSes have different ACK timing patterns
  - Linux ~200ms delayed ACK, Windows ~40ms
  - Clock granularity differs (Linux 1ms, Windows 100ms)
- **Implementation**:
  - Calculate time delta between consecutive packets
  - Requires sorting packets by timestamp within flows
- **Where to add**:
  - As a post-processing step after packet extraction
  - Group by flow (5-tuple), sort by timestamp, calculate deltas

### 4. Initial Sequence Number (ISN)
- **Status**: NOT extracted explicitly (sequence number of SYN packet)
- **Importance**: CRITICAL for OS fingerprinting
- **Use case**:
  - Windows: Uses predictable ISN (RFC 793 compliant)
  - Linux: Uses secure random ISN (RFC 6528)
  - Randomness analysis distinguishes OS families
- **Implementation**:
  - Extract `tcp.seq` from SYN packets specifically
  - Flag as ISN for SYN packets, regular seq for others
- **Where to add**: Same as sequence number, with special handling for SYN

---

## Feature Enhancement Plan

### Phase 1: Add Basic Sequence/ACK Numbers (Quick Win)

**Files to modify:**
1. `preprocess_scripts/nprint_preprocess.py`
2. `preprocess_scripts/cesnet_preprocess.py`

**Changes:**
```python
# In extract features section (around line 356-379 for nprint, 243-279 for cesnet)
# Add these fields to the record dict:

# TCP sequence and acknowledgment numbers
'tcp_seq': tcp_layer.seq if hasattr(tcp_layer, 'seq') else None,
'tcp_ack': tcp_layer.ack if hasattr(tcp_layer, 'ack') else None,

# Flag ISN for SYN packets
'is_syn': bool(tcp_layer.flags & 0x02),
'tcp_isn': tcp_layer.seq if (tcp_layer.flags & 0x02) else None,  # ISN only for SYN packets
```

**Estimated impact:**
- CRITICAL for distinguishing OS implementations
- Essential for ACK-based fingerprinting (CESNET dataset)
- Enables ISN randomness analysis

### Phase 2: Add Inter-Packet Timing (Moderate Complexity)

**Approach 1: Simple IPT (packet-to-packet)**
```python
# After extracting all packets, group by flow and calculate IPT
def calculate_ipt(df):
    """Calculate inter-packet timing within flows"""
    # Group by 5-tuple
    df = df.sort_values(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp'])

    # Calculate time delta to next packet in same flow
    df['ipt'] = df.groupby(['src_ip', 'dst_ip', 'src_port', 'dst_port'])['timestamp'].diff()

    return df
```

**Approach 2: Statistical IPT (flow-level)**
```python
# For flow-based features
'ipt_mean': Calculate mean IPT for flow
'ipt_std': Calculate std dev of IPT
'ipt_min': Minimum IPT
'ipt_max': Maximum IPT
```

**Where to add:**
- As a post-processing function in each preprocessing script
- Call after DataFrame creation, before saving to CSV

**Estimated impact:**
- HIGH for behavioral fingerprinting
- Complements static features (window size, TTL)
- Helps distinguish between similar OS versions

### Phase 3: Enhanced ISN Analysis (Advanced)

**Features to add:**
```python
# ISN randomness metrics (requires multiple SYN packets from same host)
'isn_entropy': Calculate Shannon entropy of ISN values
'isn_predicatability': Measure ISN predictability
'isn_increment_pattern': Pattern of ISN increments
```

**Implementation:**
- Requires grouping multiple flows by source IP
- Statistical analysis of ISN distribution
- Best done as a separate analysis script

**Where to add:**
- New script: `analyze_isn_patterns.py`
- Operates on preprocessed data with ISN values

---

## Inference Pipeline Recommendations

Based on your proposed pipeline:

### ✅ Your Proposed Pipeline (Validated)

```
Step 1: Capture flow (TCP connection) in defined time interval
        ↓
Step 2: Feed flow to Masaryk model → Predict OS Family
        ↓
Step 3: Route based on family:
        ├─ IF family needs SYN-based detection:
        │  └─ Extract TCP SYN packet → nprint model → Predict OS Version
        │
        └─ IF family needs ACK-based detection:
           └─ Extract 3-5 ACK packets → CESNET idle model → Predict OS Version
                ↓
Step 4: Output most confident prediction
```

### 📊 Dataset Composition Analysis

**CESNET Idle Dataset:**
- 99.9% ACK packets ✓
- 0.1% SYN packets ⚠️
- **Recommendation**: Train CESNET model on ACK packets (3-5 per flow)
- **Benefit**: MASSIVE increase in training data (1000x more samples!)

**nprint Dataset:**
- Composition unknown (need to run analysis script)
- **TODO**: Run `analyze_pcapng_composition.py` to determine SYN/ACK ratio
- **Action**: If mostly ACK, consider same approach as CESNET

### 🎯 Enhanced Pipeline with New Features

```
Step 1: Capture flow (5-10 packets minimum)
        - Extract flow-level features (duration, packet count)
        - Calculate IPT statistics
        ↓
Step 2: Masaryk model (flow-based)
        Input: Flow features (window size, TTL, MSS, DF, TOS, IPT stats)
        Output: OS Family + confidence
        ↓
Step 3: Route based on packet availability:

        IF TCP SYN available:
        ├─ Extract: TTL, window size, MSS, options order, ISN, TOS
        └─ Feed to nprint model → OS Version

        IF TCP SYN not available (or low confidence):
        ├─ Extract 3-5 ACK packets from flow
        ├─ Features: TTL, window size, seq/ack numbers, IPT, TOS, IP ID
        └─ Feed to CESNET idle model → OS Version
        ↓
Step 4: Ensemble prediction
        - Combine Family + Version predictions
        - Weight by confidence scores
        - Return: OS Family + Version + Confidence
```

### 🔑 Key Advantages of This Approach

1. **Flexibility**: Works with SYN packets OR ACK packets
2. **Data Efficiency**: Uses 99.9% of CESNET dataset (not just 0.1%)
3. **Robustness**: Multiple prediction paths increase reliability
4. **Real-world Ready**: Many networks filter SYN packets, ACK-based fallback essential

### 🚨 Important Considerations

#### Feature Availability by Packet Type

| Feature | SYN Packet | ACK Packet | Importance |
|---------|-----------|------------|------------|
| tcp_mss | ✅ | ❌ | HIGH |
| tcp_window_scale | ✅ | ❌ | HIGH |
| tcp_options_order | ✅ | ⚠️ (different) | CRITICAL |
| tcp_timestamp | ✅ | ✅ | CRITICAL |
| ISN | ✅ | ❌ | CRITICAL |
| tcp_seq/ack | ✅ | ✅ | HIGH |
| IPT | ⚠️ (limited) | ✅ | HIGH |
| ip_id pattern | ⚠️ (limited) | ✅ | HIGH |

**Key Insight**: SYN and ACK packets have DIFFERENT discriminative features!
- SYN: Options (MSS, WScale), ISN
- ACK: Sequence patterns, IPT, IP ID progression

This justifies having **separate models** for SYN-based and ACK-based fingerprinting!

---

## Implementation Priority

### 🚀 High Priority (Do First)
1. ✅ Create `analyze_pcapng_composition.py` - DONE
2. ⏳ Run analysis on nprint dataset to determine composition
3. ⏳ Add seq/ack numbers to nprint and cesnet preprocessing
4. ⏳ Add ISN extraction for SYN packets

### 🔧 Medium Priority (Do Soon)
5. ⏳ Add IPT calculation (post-processing step)
6. ⏳ Retrain CESNET model with ACK packets (use --all-tcp flag)
7. ⏳ Test inference pipeline with both SYN and ACK paths

### 🎯 Low Priority (Nice to Have)
8. ⏳ ISN entropy analysis script
9. ⏳ IP ID pattern analysis (Windows vs Linux)
10. ⏳ Enhanced ensemble prediction with confidence weighting

---

## Files to Modify

### Immediate Changes Needed

1. **preprocess_scripts/nprint_preprocess.py**
   - Add: Line ~380 (feature extraction)
   - Fields: `tcp_seq`, `tcp_ack`, `tcp_isn`, `is_syn`

2. **preprocess_scripts/cesnet_preprocess.py**
   - Add: Line ~265 (feature extraction)
   - Fields: `tcp_seq`, `tcp_ack`, `tcp_isn`, `is_syn`

3. **New file: `calculate_ipt.py`**
   - Post-processing script to add IPT features
   - Operates on existing CSV files
   - Groups by flow, calculates deltas

4. **Update training scripts**
   - Ensure new features are used in model training
   - Handle missing values (especially MSS/WScale in ACK packets)

---

## Next Steps

1. **Run packet composition analysis**:
   ```bash
   python analyze_pcapng_composition.py data/raw/nprint/dataset.pcapng --detailed
   ```

2. **Add sequence/ACK features**:
   - Modify `nprint_preprocess.py`
   - Modify `cesnet_preprocess.py`
   - Test on small sample

3. **Reprocess datasets with --all-tcp for CESNET**:
   ```bash
   python preprocess_scripts/cesnet_preprocess.py --all-tcp
   ```

4. **Calculate IPT features**:
   ```bash
   python calculate_ipt.py data/processed/cesnet.csv
   python calculate_ipt.py data/processed/nprint.csv
   ```

5. **Retrain models with new features**

6. **Implement inference pipeline**

---

## Questions to Consider

1. **Flow definition**: How to define flow boundaries in live capture?
   - Use 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol)?
   - Add timeout for flow termination (e.g., 60 seconds)?

2. **Packet sampling**: How many packets per flow?
   - SYN path: 1 packet (the SYN) sufficient
   - ACK path: 3-5 packets recommended (for IPT stats)

3. **Model selection logic**: Rules for routing to SYN vs ACK model?
   - Priority: SYN model if SYN available
   - Fallback: ACK model if no SYN within timeout
   - Hybrid: Use both and ensemble?

4. **Confidence thresholds**: When to reject prediction?
   - Masaryk confidence < 0.6 → Reject family prediction
   - Version model confidence < 0.5 → Return "Unknown version"

---

## Summary

### ✅ What We Have
- Comprehensive IP/TCP fingerprinting features
- TOS field ✓
- TCP timestamps ✓
- Flow-level statistics (Masaryk) ✓
- Packet-level features (nprint/CESNET) ✓

### ❌ What We Need
- Sequence/ACK numbers (CRITICAL for ACK-based fingerprinting)
- Inter-Packet Timing (HIGH importance for behavioral analysis)
- Initial Sequence Number (ISN) flagging (CRITICAL for OS detection)

### 🎯 Impact
Adding these features will:
1. Enable effective use of 99.9% of CESNET dataset (ACK packets)
2. Distinguish between similar OS versions (e.g., Windows 10 vs 11)
3. Provide fallback when SYN packets unavailable
4. Improve overall classification accuracy by 10-20% (estimated)

### 📊 Validation
Your proposed inference pipeline is **SOUND** and will work well:
- ✅ Flow-based family classification (Masaryk)
- ✅ SYN-based version classification (nprint)
- ✅ ACK-based version classification (CESNET idle)
- ✅ Ensemble for final prediction

The key enhancement is to **fully utilize the ACK-heavy CESNET dataset** by extracting seq/ack/IPT features!
