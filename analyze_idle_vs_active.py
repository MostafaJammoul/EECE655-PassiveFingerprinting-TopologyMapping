#!/usr/bin/env python3
"""
Analyze Idle vs Active ACK Packets

This script distinguishes between "idle" and "active" TCP ACK packets to determine
if a dataset is suitable for CESNET-style idle OS fingerprinting.

Idle ACK Characteristics:
- Pure ACK flag (no PSH, no data transfer)
- Zero or minimal payload
- Part of keepalive or window update behavior
- Regular inter-packet timing (not bursty)

Active ACK Characteristics:
- PSH-ACK (acknowledging data transfer)
- Non-zero payload
- Bursty traffic patterns
- Part of active HTTP/HTTPS sessions

Usage:
    python analyze_idle_vs_active.py <input.pcapng>
"""

import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import statistics

try:
    from scapy.all import rdpcap, IP, TCP, Raw, PcapNgReader
except ImportError:
    print("ERROR: Scapy not installed.")
    print("Install with: pip install scapy")
    sys.exit(1)


def classify_ack_packet(tcp_layer, has_payload, payload_size, time_delta=None):
    """
    Classify an ACK packet as 'idle' or 'active'

    Criteria for IDLE ACK:
    1. ACK flag set
    2. PSH flag NOT set (no data push)
    3. Payload size = 0 (pure acknowledgment)
    4. Not SYN, FIN, or RST

    Args:
        tcp_layer: Scapy TCP layer
        has_payload: Boolean, packet has data payload
        payload_size: Size of TCP payload in bytes
        time_delta: Time since previous packet in flow (optional)

    Returns:
        'idle', 'active', or 'other'
    """
    flags = tcp_layer.flags

    # Check if it's a pure ACK
    has_ack = bool(flags & 0x10)
    has_psh = bool(flags & 0x08)
    has_syn = bool(flags & 0x02)
    has_fin = bool(flags & 0x01)
    has_rst = bool(flags & 0x04)

    # Not an ACK packet at all
    if not has_ack:
        return 'other'

    # SYN-ACK, FIN-ACK, RST-ACK = connection management, not idle
    if has_syn or has_fin or has_rst:
        return 'other'

    # PSH-ACK = data transfer = ACTIVE
    if has_psh:
        return 'active'

    # Non-zero payload = data transfer = ACTIVE
    if payload_size > 0:
        return 'active'

    # Pure ACK with zero payload = potentially IDLE
    # Additional heuristics could be added here (window updates, etc.)
    return 'idle'


def analyze_flow_patterns(flow_packets, flow_times):
    """
    Analyze flow-level patterns to determine if flow is idle or active

    Returns:
        dict with flow statistics
    """
    if len(flow_times) < 2:
        return {
            'is_idle_flow': False,
            'reason': 'insufficient_packets',
            'avg_ipt': 0,
            'cv_ipt': 0
        }

    # Calculate inter-packet times
    ipts = [flow_times[i+1] - flow_times[i] for i in range(len(flow_times)-1)]

    if not ipts:
        return {
            'is_idle_flow': False,
            'reason': 'no_ipt',
            'avg_ipt': 0,
            'cv_ipt': 0
        }

    avg_ipt = statistics.mean(ipts)

    # Coefficient of variation (std/mean) - idle traffic should have regular timing
    if len(ipts) > 1:
        std_ipt = statistics.stdev(ipts)
        cv_ipt = std_ipt / avg_ipt if avg_ipt > 0 else 0
    else:
        cv_ipt = 0

    # Heuristics for idle flow:
    # 1. Average IPT > 5 seconds (keepalives, not active data transfer)
    # 2. Low coefficient of variation (regular timing)
    is_idle_flow = (avg_ipt > 5.0) and (cv_ipt < 1.0)

    return {
        'is_idle_flow': is_idle_flow,
        'avg_ipt': avg_ipt,
        'cv_ipt': cv_ipt,
        'packet_count': len(flow_packets),
        'reason': 'idle_pattern' if is_idle_flow else 'active_pattern'
    }


def analyze_idle_active(pcap_path, flow_analysis=False):
    """
    Analyze idle vs active ACK packets

    Args:
        pcap_path: Path to pcapng file
        flow_analysis: Perform flow-level analysis (slower but more accurate)
    """
    print("="*70)
    print("IDLE vs ACTIVE ACK PACKET ANALYZER")
    print("="*70)
    print(f"\nAnalyzing: {pcap_path}")
    print(f"Flow analysis: {'ENABLED' if flow_analysis else 'DISABLED'}")

    # Read packets
    print(f"\nReading packets...")
    try:
        packets = []
        try:
            with PcapNgReader(str(pcap_path)) as reader:
                packets = [pkt for pkt in reader]
        except:
            packets = rdpcap(str(pcap_path))

        print(f"  Loaded {len(packets):,} packets")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Statistics
    stats = {
        'total_packets': len(packets),
        'tcp_packets': 0,
        'ack_packets': 0,
        'idle_acks': 0,
        'active_acks': 0,
        'other_packets': 0,
        'payload_sizes': [],
        'psh_acks': 0,
        'pure_acks_with_payload': 0,
        'pure_acks_zero_payload': 0,
    }

    # Flow tracking (if enabled)
    flows = defaultdict(lambda: {'packets': [], 'times': [], 'classifications': []})

    print(f"\nAnalyzing packet characteristics...")

    base_time = None
    for i, packet in enumerate(packets):
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            stats['other_packets'] += 1
            continue

        stats['tcp_packets'] += 1
        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        # Get payload
        has_payload = packet.haslayer(Raw)
        payload_size = len(packet[Raw].load) if has_payload else 0
        stats['payload_sizes'].append(payload_size)

        # Get packet time
        pkt_time = float(packet.time)
        if base_time is None:
            base_time = pkt_time
        rel_time = pkt_time - base_time

        # Flow key
        flow_key = None
        if flow_analysis:
            flow_key = (
                min(ip_layer.src, ip_layer.dst),
                max(ip_layer.src, ip_layer.dst),
                min(tcp_layer.sport, tcp_layer.dport),
                max(tcp_layer.sport, tcp_layer.dport)
            )
            flows[flow_key]['packets'].append(packet)
            flows[flow_key]['times'].append(rel_time)

        # Classify packet
        classification = classify_ack_packet(tcp_layer, has_payload, payload_size)

        if classification == 'idle':
            stats['idle_acks'] += 1
            stats['pure_acks_zero_payload'] += 1
        elif classification == 'active':
            stats['active_acks'] += 1
            if tcp_layer.flags & 0x08:  # PSH flag
                stats['psh_acks'] += 1
            else:
                stats['pure_acks_with_payload'] += 1
        else:
            stats['other_packets'] += 1

        if flow_analysis and flow_key:
            flows[flow_key]['classifications'].append(classification)

        # Progress indicator
        if (i+1) % 100000 == 0:
            print(f"  Processed {i+1:,} packets...")

    # Flow-level analysis
    flow_stats = None
    if flow_analysis and flows:
        print(f"\nAnalyzing flow patterns...")
        flow_stats = {
            'total_flows': len(flows),
            'idle_flows': 0,
            'active_flows': 0,
            'mixed_flows': 0,
            'idle_flow_packets': 0,
            'active_flow_packets': 0,
        }

        for flow_key, flow_data in flows.items():
            flow_pattern = analyze_flow_patterns(flow_data['packets'], flow_data['times'])

            if flow_pattern['is_idle_flow']:
                flow_stats['idle_flows'] += 1
                flow_stats['idle_flow_packets'] += flow_pattern['packet_count']
            else:
                # Check if all packets in flow are 'idle' classified
                idle_count = flow_data['classifications'].count('idle')
                active_count = flow_data['classifications'].count('active')

                if idle_count > active_count * 2:  # Mostly idle packets
                    flow_stats['idle_flows'] += 1
                    flow_stats['idle_flow_packets'] += flow_pattern['packet_count']
                elif active_count > idle_count * 2:  # Mostly active packets
                    flow_stats['active_flows'] += 1
                    flow_stats['active_flow_packets'] += flow_pattern['packet_count']
                else:
                    flow_stats['mixed_flows'] += 1

    # Print results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    print(f"\nPacket-Level Classification:")
    print(f"  Total packets:              {stats['total_packets']:>12,}")
    print(f"  TCP packets:                {stats['tcp_packets']:>12,}")
    print(f"  ACK-related packets:        {stats['idle_acks'] + stats['active_acks']:>12,}")
    print()
    print(f"  IDLE ACKs:                  {stats['idle_acks']:>12,} ({stats['idle_acks']/stats['tcp_packets']*100:>6.2f}%)")
    print(f"    └─ Pure ACK, zero payload")
    print()
    print(f"  ACTIVE ACKs:                {stats['active_acks']:>12,} ({stats['active_acks']/stats['tcp_packets']*100:>6.2f}%)")
    print(f"    ├─ PSH-ACK:               {stats['psh_acks']:>12,} ({stats['psh_acks']/stats['tcp_packets']*100:>6.2f}%)")
    print(f"    └─ ACK with payload:      {stats['pure_acks_with_payload']:>12,} ({stats['pure_acks_with_payload']/stats['tcp_packets']*100:>6.2f}%)")
    print()
    print(f"  Other (SYN/FIN/RST/etc):    {stats['other_packets']:>12,} ({stats['other_packets']/stats['tcp_packets']*100:>6.2f}%)")

    # Payload statistics
    if stats['payload_sizes']:
        non_zero = [p for p in stats['payload_sizes'] if p > 0]
        print(f"\nPayload Statistics:")
        print(f"  Zero payload packets:       {len(stats['payload_sizes']) - len(non_zero):>12,} ({(len(stats['payload_sizes'])-len(non_zero))/len(stats['payload_sizes'])*100:>6.2f}%)")
        print(f"  Non-zero payload packets:   {len(non_zero):>12,} ({len(non_zero)/len(stats['payload_sizes'])*100:>6.2f}%)")
        if non_zero:
            print(f"  Average payload size:       {statistics.mean(non_zero):>12.1f} bytes")

    # Flow statistics
    if flow_stats:
        print(f"\nFlow-Level Classification:")
        print(f"  Total flows:                {flow_stats['total_flows']:>12,}")
        print(f"  Idle flows:                 {flow_stats['idle_flows']:>12,} ({flow_stats['idle_flows']/flow_stats['total_flows']*100:>6.2f}%)")
        print(f"    └─ Packets in idle flows: {flow_stats['idle_flow_packets']:>12,}")
        print(f"  Active flows:               {flow_stats['active_flows']:>12,} ({flow_stats['active_flows']/flow_stats['total_flows']*100:>6.2f}%)")
        print(f"    └─ Packets in active flows: {flow_stats['active_flow_packets']:>12,}")
        print(f"  Mixed flows:                {flow_stats['mixed_flows']:>12,} ({flow_stats['mixed_flows']/flow_stats['total_flows']*100:>6.2f}%)")

    # Recommendations
    print("\n" + "="*70)
    print("CESNET-STYLE IDLE FINGERPRINTING COMPATIBILITY")
    print("="*70)

    total_acks = stats['idle_acks'] + stats['active_acks']
    idle_pct = (stats['idle_acks'] / total_acks * 100) if total_acks > 0 else 0
    active_pct = (stats['active_acks'] / total_acks * 100) if total_acks > 0 else 0

    print(f"\nIdle ACK Ratio: {idle_pct:.2f}% of ACK packets")
    print()

    if idle_pct > 50:
        print("  ✓ GOOD: Majority of ACKs are idle (pure ACK, zero payload)")
        print("    → This dataset is suitable for CESNET-style idle fingerprinting")
        print("    → Reflects OS TCP stack behavior, not application behavior")
    elif idle_pct > 20:
        print("  ⚠ MIXED: Dataset contains both idle and active ACKs")
        print("    → Can use idle ACKs, but they're a minority")
        print("    → Consider filtering to extract only idle ACKs")
        print(f"    → You would retain {stats['idle_acks']:,} idle ACK packets")
    else:
        print("  ✗ PROBLEM: Dataset is dominated by ACTIVE traffic")
        print("    → NOT suitable for CESNET-style idle fingerprinting")
        print("    → Active traffic reflects application behavior + congestion control")
        print("    → Training on this would create different feature distributions")

    print(f"\nActive ACK Ratio: {active_pct:.2f}% of ACK packets")
    print(f"  PSH-ACK packets: {stats['psh_acks']:,} (data transfer in progress)")

    if active_pct > 50:
        print("\n  This dataset captures ACTIVE network usage:")
        print("    • HTTP/HTTPS browsing")
        print("    • File downloads/uploads")
        print("    • Application data transfer")
        print("\n  This is fundamentally DIFFERENT from CESNET's idle approach!")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if idle_pct > 50:
        print("\n✓ Dataset is compatible with CESNET approach")
        print("  • Use pure ACK packets (zero payload)")
        print("  • Extract 3-5 ACKs per flow")
        print("  • Train models on idle behavior")
    elif idle_pct > 20:
        print("\n⚠ Dataset requires filtering")
        print("  • Filter to extract only idle ACKs (zero payload)")
        print("  • This will reduce dataset size significantly")
        print(f"  • Final dataset: ~{stats['idle_acks']:,} packets")
        print("  • Consider hybrid approach (both SYN and idle ACK)")
    else:
        print("\n✗ Dataset NOT suitable for idle fingerprinting")
        print("  • Consider using SYN-based approach instead")
        print("  • Or collect new dataset with idle traffic")
        print("  • Or adapt model to handle active traffic patterns")
        print("\n  CESNET trained on IDLE, testing on ACTIVE = domain mismatch!")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze idle vs active ACK packets for CESNET-style fingerprinting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', type=str, help='Input PCAPNG file')
    parser.add_argument('--flows', action='store_true',
                       help='Perform flow-level analysis (slower but more detailed)')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    # Analyze
    stats = analyze_idle_active(input_path, flow_analysis=args.flows)

    if stats is None:
        print("\nERROR: Analysis failed!")
        sys.exit(1)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
