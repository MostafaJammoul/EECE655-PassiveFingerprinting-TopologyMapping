# Idle vs Active Traffic: Critical Analysis for OS Fingerprinting

## The Problem

**CESNET's approach** relies on capturing TCP packets when machines are **idle** (no active user applications).
**Your nprint dataset** appears to contain **active** traffic (web browsing, downloads).

This creates a **domain mismatch** problem that could severely impact model performance.

---

## What "Idle" Actually Means

### CESNET Idle Traffic
```
✓ No active user applications running
✓ OS background services only:
  - TCP keepalive packets (every 60-120 seconds)
  - Windows/Linux update checks
  - Time synchronization (NTP → generates periodic TCP)
  - DNS cache maintenance
  - Background telemetry

✓ TCP ACK characteristics:
  - Zero payload (pure acknowledgments)
  - Regular inter-packet timing
  - Default window sizes (no dynamic scaling)
  - Minimal TCP options
  - Reflects OS TCP STACK behavior, not applications
```

### Active Traffic (What Your nprint Dataset Likely Contains)
```
✗ Active user applications (browsers, downloads, streaming)
✗ TCP ACK characteristics:
  - PSH-ACK flags (20.25% in your dataset!)
  - Non-zero payloads (data transfer)
  - Bursty, irregular timing
  - Dynamic window scaling (reacting to throughput)
  - Application-influenced TCP options
  - Reflects APPLICATION behavior + OS + network conditions
```

---

## Evidence from Your Analysis

### Your Dataset Statistics:
```
TCP Packet Type Distribution:
  ACK:         865,739  (69.60%)  ← Could be idle OR active
  PSH-ACK:     251,902  (20.25%)  ← DEFINITELY ACTIVE (data transfer!)
  FIN-ACK:      52,425  (4.21%)
  SYN:          48,128  (3.87%)
```

### Key Indicators of ACTIVE Traffic:

1. **20.25% PSH-ACK packets**
   - PSH flag = "push data to application layer immediately"
   - Only set during active data transfer
   - **Idle traffic would have ~0% PSH-ACK**

2. **86% HTTPS/HTTP traffic** (port 443: 55.49%, port 80: 30.02%)
   - Web browsing, API calls, downloads
   - NOT idle background services

3. **High destination diversity**
   - Microsoft servers (13.107.4.50)
   - CDN servers (23.15.4.9, 72.21.91.29)
   - Indicates active web requests

4. **Outbound-heavy** (SYN >> SYN-ACK: 8.4:1)
   - Client machines initiating connections
   - Browsing behavior, not server idle state

---

## The Domain Mismatch Problem

### If You Train on CESNET (Idle) and Test on nprint (Active):

| Feature | CESNET (Idle) | nprint (Active) | Impact |
|---------|---------------|-----------------|--------|
| **Window Size** | Static, default values | Dynamic, scaled for throughput | ❌ Distribution shift |
| **TCP Options** | Default OS options | Negotiated for performance | ❌ Option patterns differ |
| **IPT (Inter-Packet Time)** | Regular (60s, 120s intervals) | Bursty (ms-level clusters) | ❌ Timing features invalid |
| **Payload Size** | 0 bytes | Variable (0 to 1460 bytes) | ❌ Payload stats differ |
| **Sequence Numbers** | Minimal growth | Rapid growth (data transfer) | ❌ Seq/ACK patterns differ |
| **PSH Flag** | ~0% | 20.25% | ❌ Flag distribution shift |

**Result:** Model learns to classify **OS+application+network** behavior, not pure OS behavior.

---

## How to Distinguish Idle from Active ACKs

### Run the Analysis Script:
```bash
python analyze_idle_vs_active.py data/raw/nprint/os-100-packet.pcapng
```

### Criteria for "Idle" ACK:
```python
✓ ACK flag is SET
✓ PSH flag is NOT SET  ← No data push
✓ Payload size = 0     ← Pure acknowledgment
✓ NOT (SYN or FIN or RST)
```

### Expected Results:

**If nprint is truly idle (like CESNET):**
```
Idle ACKs:   > 80% of ACK packets
Active ACKs: < 20% of ACK packets
✓ Compatible with CESNET approach
```

**If nprint is active (likely):**
```
Idle ACKs:   < 30% of ACK packets
Active ACKs: > 50% of ACK packets  ← PSH-ACK dominates
✗ NOT compatible with CESNET approach
```

---

## Verification: Run These Checks

### 1. Check Payload Distribution
```bash
# Extract payload sizes
tshark -r data/raw/nprint/os-100-packet.pcapng \
  -Y "tcp.flags.ack==1 && !tcp.flags.syn && !tcp.flags.fin && !tcp.flags.rst" \
  -T fields -e tcp.len | \
  awk '{if($1==0) zero++; else nonzero++} END {
    print "Zero payload:", zero;
    print "Non-zero payload:", nonzero;
    print "Zero %:", (zero/(zero+nonzero)*100)
  }'
```

**Idle traffic:** >80% zero payload
**Active traffic:** <50% zero payload

### 2. Check PSH Flag on ACKs
```bash
# Count PSH-ACK vs pure ACK
tshark -r data/raw/nprint/os-100-packet.pcapng \
  -Y "tcp.flags.ack==1 && !tcp.flags.syn" \
  -T fields -e tcp.flags.push | \
  awk '{if($1==1) psh++; else nopsh++} END {
    print "PSH-ACK:", psh;
    print "Pure ACK:", nopsh;
    print "PSH %:", (psh/(psh+nopsh)*100)
  }'
```

**Idle traffic:** <5% PSH-ACK
**Active traffic:** >15% PSH-ACK (you have 20.25%!)

### 3. Check Inter-Packet Timing
```bash
# Check if packets are bursty (active) or regular (idle)
python analyze_idle_vs_active.py --flows data/raw/nprint/os-100-packet.pcapng
```

This will show flow-level timing patterns.

---

## Implications for Your Project

### Option 1: Accept Domain Mismatch (NOT RECOMMENDED)
- Train on CESNET idle traffic
- Test on nprint active traffic
- **Result:** Poor accuracy, unreliable results
- **Risk:** Conclusions are invalid

### Option 2: Filter nprint to Extract Idle ACKs
```bash
# Extract only idle-like ACKs (zero payload, no PSH)
tshark -r data/raw/nprint/os-100-packet.pcapng \
  -Y "tcp.flags.ack==1 && !tcp.flags.push && tcp.len==0 && \
      !tcp.flags.syn && !tcp.flags.fin && !tcp.flags.rst" \
  -w data/processed/nprint_idle_only.pcapng
```

**Pros:**
- Creates comparable dataset
- Valid CESNET comparison

**Cons:**
- Drastically reduces dataset size
- May not have enough samples per OS

### Option 3: Use SYN-Based Fingerprinting (RECOMMENDED)
- Ignore CESNET's ACK approach
- Use traditional SYN-based fingerprinting (48,128 SYN packets)
- SYN packets are standardized (independent of idle/active state)

**Pros:**
- No domain mismatch issue
- 48K SYN packets sufficient
- Well-established approach (p0f, nmap)

**Cons:**
- Cannot replicate CESNET's ACK approach
- Fewer samples than ACK-based (48K vs 865K)

### Option 4: Collect New Idle Dataset
- Set up VMs with 10 different OS versions
- Let them idle for 30 minutes each
- Capture background TCP traffic
- Use this for CESNET-style approach

**Pros:**
- Ground truth idle traffic
- Perfect comparison with CESNET

**Cons:**
- Time-consuming
- Requires VM setup and network capture

---

## Recommended Approach for EECE655 Project

### Phase 1: Verify Traffic Type
```bash
python analyze_idle_vs_active.py data/raw/nprint/os-100-packet.pcapng
```

### Phase 2: Decision Tree

**If Idle ACKs > 60%:**
- ✓ Use CESNET ACK-based approach
- ✓ Extract idle ACKs for training/testing
- ✓ Compare with CESNET methodology

**If Idle ACKs < 30% (LIKELY):**
- ✗ Abandon ACK-based approach for nprint
- ✓ Use SYN-based approach instead
- ✓ Document this as a limitation
- ✓ Explain why domain mismatch is problematic

### Phase 3: Implementation
```
For Your Pipeline:

1. CESNET Dataset → ACK-based Model
   └─ Train/test on CESNET idle traffic only

2. nprint Dataset → SYN-based Model
   └─ Train/test on nprint SYN packets only

3. Comparison Study:
   ├─ CESNET (Idle ACK) vs nprint (SYN) accuracy
   ├─ Document traffic type differences
   └─ Explain why direct comparison is invalid
```

---

## Key Takeaway

**You CANNOT directly compare CESNET (idle) with nprint (active) using the same approach.**

It's like training a face recognition model on frontal faces, then testing on profile faces.
The features are fundamentally different.

Either:
1. Filter nprint to extract idle-like packets
2. Use different approaches for each dataset (ACK for CESNET, SYN for nprint)
3. Collect new idle traffic for nprint

**Document this methodological choice clearly in your report!**

---

## Next Steps

1. Run `python analyze_idle_vs_active.py` on your nprint dataset
2. Check the idle/active ratio
3. Make informed decision based on results
4. Update your pipeline accordingly
5. Document traffic type assumptions in your methodology
