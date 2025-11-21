# Complete Feature Extraction Guide

## Overview

The `extract_all_ack_features.py` script extracts **ALL 31 features** from passive/idle ACK packets in both nprint and CESNET datasets.

## Features Extracted (31 Total)

### 1. Metadata (5 features)
- `record_id`: Unique packet identifier
- `dataset_source`: Source dataset (nprint/cesnet)
- `timestamp`: Packet timestamp
- `packet_type`: Classification (idle_ack, active_ack, psh_ack, syn, fin_ack, etc.)
- `label`: OS label (from PCAPNG comment or directory structure)

### 2. IP Layer (8 features)
- `src_ip`: Source IP address
- `dst_ip`: Destination IP address
- `ttl`: Time To Live (observed)
- `initial_ttl`: Inferred initial TTL (32/64/128/255)
- `df_flag`: Don't Fragment flag (0/1)
- `ip_id`: IP Identification field
- `ip_tos`: Type of Service
- `ip_len`: Total IP packet length

### 3. TCP Layer (10 features)
- `src_port`: Source port
- `dst_port`: Destination port
- `window_size`: TCP window size
- `mss`: Maximum Segment Size (from options)
- `window_scale`: Window scale factor (from options)
- `options_order`: Order of TCP options (e.g., "2,4,8,1,3")
- `timestamp_val`: TCP timestamp value
- `timestamp_ecr`: TCP timestamp echo reply
- `sack_permitted`: SACK permitted flag (0/1)
- `tcp_flags`: TCP flags (numeric)
- `urgent_ptr`: Urgent pointer
- `payload_len`: Payload length in bytes

### 4. TCP Sequence (3 features)
- `tcp_seq`: Sequence number
- `tcp_ack`: Acknowledgment number
- `tcp_isn`: Initial Sequence Number (from flow's SYN packet)

### 5. IPT Features (7 features)
- `ipt`: Inter-Packet Time from previous packet in flow
- `ipt_next`: Inter-Packet Time to next packet in flow
- `ipt_mean`: Mean IPT for the flow
- `ipt_std`: Standard deviation of IPT for the flow
- `ipt_min`: Minimum IPT in the flow
- `ipt_max`: Maximum IPT in the flow
- `ipt_median`: Median IPT for the flow

## Usage Examples

### 1. Extract from nprint dataset (ACTIVE traffic)

```bash
# Extract ALL TCP packets
python extract_all_ack_features.py \
    --dataset nprint \
    --input data/raw/nprint/os-100-packet.pcapng \
    --output features_nprint_all.csv

# Extract only IDLE ACKs (zero payload, no PSH)
python extract_all_ack_features.py \
    --dataset nprint \
    --input data/raw/nprint/os-100-packet.pcapng \
    --output features_nprint_idle.csv \
    --idle-only

# Extract all ACK packets (idle + active + PSH-ACK)
python extract_all_ack_features.py \
    --dataset nprint \
    --input data/raw/nprint/os-100-packet.pcapng \
    --output features_nprint_ack.csv \
    --all-ack
```

### 2. Extract from CESNET dataset (IDLE traffic)

```bash
# Extract ALL TCP packets
python extract_all_ack_features.py \
    --dataset cesnet \
    --input data/raw/cesnet \
    --output features_cesnet_all.csv

# Extract only IDLE ACKs
python extract_all_ack_features.py \
    --dataset cesnet \
    --input data/raw/cesnet \
    --output features_cesnet_idle.csv \
    --idle-only

# Extract all ACK packets
python extract_all_ack_features.py \
    --dataset cesnet \
    --input data/raw/cesnet \
    --output features_cesnet_ack.csv \
    --all-ack
```

### 3. Quick Test (sample data)

```bash
# Test on nprint dataset
python extract_all_ack_features.py \
    --dataset nprint \
    --input data/raw/nprint/os-100-packet.pcapng \
    --output test_nprint.csv \
    --idle-only
```

## Packet Type Classification

The script automatically classifies packets into these types:

| Type | Description | Criteria |
|------|-------------|----------|
| `idle_ack` | **Pure idle ACK** | ACK flag set, zero payload, no PSH |
| `active_ack` | ACK with data | ACK flag set, non-zero payload, no PSH |
| `psh_ack` | PSH-ACK | ACK + PSH flags, usually with payload |
| `syn` | SYN packet | SYN flag (connection initiation) |
| `syn_ack` | SYN-ACK | SYN + ACK flags |
| `fin_ack` | FIN-ACK | FIN + ACK flags (connection termination) |
| `rst` | RST packet | RST flag (connection reset) |
| `other` | Other TCP | Any other combination |

## Filtering Options

### `--idle-only`
Extracts **ONLY** pure idle ACK packets:
- ✓ Zero payload
- ✓ ACK flag set
- ✓ No PSH flag
- **Use this for CESNET-style idle fingerprinting**

### `--all-ack`
Extracts **ALL** ACK-related packets:
- ✓ Idle ACKs (pure ACK, zero payload)
- ✓ Active ACKs (ACK with data, no PSH)
- ✓ PSH-ACKs (ACK + PSH, usually with data)
- **Use this for comprehensive ACK analysis**

### Neither flag (default)
Extracts **ALL** TCP packets (including SYN, FIN, RST, etc.)

## Output Format

The script generates a CSV file with 31 columns in this order:

```
record_id, dataset_source, timestamp, packet_type, label,
src_ip, dst_ip, ttl, initial_ttl, df_flag, ip_id, ip_tos, ip_len,
src_port, dst_port, window_size, mss, window_scale, options_order,
timestamp_val, timestamp_ecr, sack_permitted, tcp_flags, urgent_ptr, payload_len,
tcp_seq, tcp_ack, tcp_isn,
ipt, ipt_next, ipt_mean, ipt_std, ipt_min, ipt_max, ipt_median
```

## Expected Output

### nprint dataset
```
======================================================================
NPRINT DATASET FEATURE EXTRACTION
======================================================================

Input:  data/raw/nprint/os-100-packet.pcapng
Output: features_nprint_idle.csv
Mode:   Idle ACKs only

Reading packets...
  Loaded 1,243,900 packets

Identifying flows and ISNs...
  Found 12,439 flows

Extracting features...
  Processed 100,000 packets...
  Processed 200,000 packets...
  ...

Calculating IPT features...

Writing 829,005 records to CSV...
  Wrote 829,005 rows × 31 columns

======================================================================
EXTRACTION COMPLETE
======================================================================

Total packets extracted: 829,005
Output file: features_nprint_idle.csv

Packet Type Distribution:
  idle_ack            :  829,005 (100.00%)
```

### CESNET dataset
```
======================================================================
CESNET DATASET FEATURE EXTRACTION
======================================================================

Input:  data/raw/cesnet
Output: features_cesnet_idle.csv
Mode:   Idle ACKs only

Scanning for PCAP files...
  Found 42 PCAP files

[1/42] Processing: traffic.pcap
  Label: Debian 12
  Loaded 12,345 packets
  Extracted 11,987 packets

[2/42] Processing: traffic.pcap
  Label: Ubuntu 22.04 LTS
  ...

Writing 504,321 records to CSV...
  Wrote 504,321 rows × 31 columns

======================================================================
EXTRACTION COMPLETE
======================================================================

Total packets extracted: 504,321
Total files processed: 42
Output file: features_cesnet_idle.csv

Packet Type Distribution:
  idle_ack            :  504,321 (100.00%)

OS Labels: 42 unique labels
  Ubuntu 22.04 LTS          :   45,123 ( 8.95%)
  Debian 12                 :   38,456 ( 7.63%)
  ...
```

## Dataset Differences

### nprint Dataset
- **Traffic Type**: ACTIVE (web browsing)
- **Label Source**: PCAPNG packet comments
- **Label Format**: `os_family_os_version` (e.g., "Linux_Ubuntu_22.04")
- **Idle ACK Ratio**: ~74% of ACK packets
- **Best Practice**: Use `--idle-only` to filter for idle behavior

### CESNET Dataset
- **Traffic Type**: IDLE (background services only)
- **Label Source**: Directory structure
- **Label Format**: `os-family_os-name_version` (e.g., "Debian 12")
- **Idle ACK Ratio**: ~99.9% of all packets
- **Best Practice**: Can use default (all TCP) or `--idle-only`

## Integration with Existing Pipeline

This script replaces/unifies:
- `preprocess_scripts/nprint_preprocess.py`
- `preprocess_scripts/cesnet_preprocess.py`
- `calculate_ipt.py`

The output CSV can be directly used for:
1. Machine learning model training
2. Statistical analysis
3. Comparison between idle vs active traffic
4. OS fingerprinting research

## Performance Notes

- **nprint dataset** (1.2M packets): ~30-60 seconds
- **CESNET dataset** (42 files, ~500K packets): ~2-5 minutes
- Memory usage: ~500MB - 1GB depending on dataset size
- IPT calculation: O(n log n) per flow due to sorting

## Troubleshooting

### "No PCAP files found"
- Check that CESNET directory structure is correct
- Expected: `data/raw/cesnet/os-family_os-name_version/*/traffic.pcap`

### "ERROR reading PCAP"
- Install scapy: `pip install scapy`
- Check file permissions
- Verify PCAP file is not corrupted

### Missing labels
- **nprint**: Ensure PCAPNG comments are preserved
- **CESNET**: Check directory naming follows convention

### IPT features are None
- Need at least 2 packets per flow
- Single-packet flows have no IPT

## Advanced Usage

### Combine both datasets

```bash
# Extract idle ACKs from both datasets
python extract_all_ack_features.py \
    --dataset nprint \
    --input data/raw/nprint/os-100-packet.pcapng \
    --output features_nprint_idle.csv \
    --idle-only

python extract_all_ack_features.py \
    --dataset cesnet \
    --input data/raw/cesnet \
    --output features_cesnet_idle.csv \
    --idle-only

# Merge in Python
import pandas as pd
df1 = pd.read_csv('features_nprint_idle.csv')
df2 = pd.read_csv('features_cesnet_idle.csv')
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv('features_combined_idle.csv', index=False)
```

### Filter by specific OS

```bash
# Extract, then filter in Python
import pandas as pd
df = pd.read_csv('features_nprint_idle.csv')
df_ubuntu = df[df['label'].str.contains('Ubuntu', na=False)]
df_ubuntu.to_csv('features_ubuntu_only.csv', index=False)
```

### Analyze specific flows

```python
import pandas as pd
df = pd.read_csv('features_nprint_idle.csv')

# Group by flow
df['flow_key'] = df['src_ip'] + ':' + df['src_port'].astype(str) + '-' + \
                 df['dst_ip'] + ':' + df['dst_port'].astype(str)

# Get flows with most packets
flow_counts = df['flow_key'].value_counts()
print(flow_counts.head(10))

# Analyze IPT for a specific flow
flow_data = df[df['flow_key'] == flow_counts.index[0]]
print(flow_data[['timestamp', 'ipt', 'ipt_mean', 'ipt_std']])
```

## Next Steps

After feature extraction:

1. **Data cleaning**: Handle missing values, outliers
2. **Feature selection**: Choose relevant features for your model
3. **Train/test split**: Separate data for validation
4. **Model training**: Use extracted features for ML
5. **Evaluation**: Compare idle vs active traffic performance

## Questions?

For issues or questions about this script, please refer to:
- Main documentation in the repository
- Original preprocessing scripts in `preprocess_scripts/`
- Analysis scripts: `analyze_idle_vs_active.py`, `analyze_pcapng_composition.py`
