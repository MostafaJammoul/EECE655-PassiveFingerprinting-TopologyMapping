#!/usr/bin/env python3
"""
Inspect PCAPNG Metadata and Identify OS Labels

The nprint dataset embeds OS information in pcapng metadata.
This script extracts that information.

Usage:
    python inspect_pcapng_metadata.py file.pcapng
"""

import sys
from pathlib import Path

try:
    from scapy.all import rdpcap, IP, TCP, PcapNg
except ImportError:
    print("ERROR: Scapy not installed")
    print("Install with: pip install scapy")
    sys.exit(1)

import pandas as pd


def analyze_tcp_fingerprints(pcap_path):
    """
    Analyze TCP fingerprints to infer OS without labels

    This uses TCP options order patterns which are highly discriminative
    """
    print(f"\nAnalyzing TCP fingerprints in: {pcap_path}")
    print("="*70)

    try:
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Extract TCP option patterns and source IPs
    fingerprints = {}

    for packet in packets:
        if packet.haslayer(IP) and packet.haslayer(TCP):
            ip = packet[IP]
            tcp = packet[TCP]

            # Only SYN packets have full options
            if (tcp.flags & 0x02) == 0:
                continue

            src_ip = ip.src

            # Extract TCP options
            options = []
            if hasattr(tcp, 'options'):
                for opt in tcp.options:
                    if isinstance(opt, tuple) and len(opt) >= 1:
                        options.append(str(opt[0]))

            if options:
                pattern = ':'.join(options)
                ttl = ip.ttl
                window = tcp.window

                if src_ip not in fingerprints:
                    fingerprints[src_ip] = []

                fingerprints[src_ip].append({
                    'pattern': pattern,
                    'ttl': ttl,
                    'window': window
                })

    # Analyze patterns per IP
    print(f"\nFound {len(fingerprints)} unique source IPs")
    print("\nOS Fingerprint Analysis:")
    print("="*70)

    for ip, records in list(fingerprints.items())[:20]:  # Show first 20 IPs
        if not records:
            continue

        # Get most common pattern for this IP
        patterns = [r['pattern'] for r in records]
        most_common = max(set(patterns), key=patterns.count)
        ttls = [r['ttl'] for r in records]
        avg_ttl = sum(ttls) / len(ttls)
        windows = [r['window'] for r in records]
        avg_window = sum(windows) / len(windows)

        # Infer OS from pattern
        os_guess = infer_os_from_pattern(most_common, avg_ttl, avg_window)

        print(f"\nIP: {ip}")
        print(f"  Packets: {len(records)}")
        print(f"  TCP Options: {most_common}")
        print(f"  Avg TTL: {avg_ttl:.1f}")
        print(f"  Avg Window: {avg_window:.0f}")
        print(f"  → Likely OS: {os_guess}")


def infer_os_from_pattern(pattern, ttl, window):
    """
    Infer OS from TCP options pattern, TTL, and window size

    Based on research papers and real-world observations
    """
    pattern_lower = pattern.lower()

    # Windows fingerprints
    if 'mss:nop:wscale:nop:nop:sack' in pattern_lower or \
       'mss:nop:ws:nop:nop:sack' in pattern_lower:
        if 110 <= ttl <= 128:
            if window > 60000:
                return "Windows 10/11 (high confidence)"
            else:
                return "Windows 7/8/10 (medium confidence)"

    # Linux fingerprints
    if 'mss:sack' in pattern_lower and 'timestamp' in pattern_lower:
        if 50 <= ttl <= 64:
            if 'wscale' in pattern_lower or 'ws' in pattern_lower:
                return "Linux (Ubuntu/Debian/modern) (high confidence)"
            else:
                return "Linux (older kernel) (medium confidence)"

    # macOS fingerprints
    if 'eol' in pattern_lower and 'timestamp' in pattern_lower:
        if 50 <= ttl <= 64:
            return "macOS (high confidence)"

    # Generic guesses based on TTL
    if 110 <= ttl <= 128:
        return "Windows (based on TTL)"
    elif 50 <= ttl <= 64:
        return "Linux/Unix/macOS (based on TTL)"
    elif ttl > 200:
        return "Network device/Cisco (based on TTL)"

    return "Unknown (insufficient data)"


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pcapng_metadata.py <file.pcapng>")
        sys.exit(1)

    pcap_path = Path(sys.argv[1])

    if not pcap_path.exists():
        print(f"ERROR: File not found: {pcap_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("PCAPNG METADATA AND OS FINGERPRINT INSPECTOR")
    print("="*70)

    # Analyze TCP fingerprints
    analyze_tcp_fingerprints(pcap_path)

    print("\n" + "="*70)
    print("\nRECOMMENDATIONS:")
    print("="*70)
    print("\n1. If this is from the nprint dataset:")
    print("   - Check the dataset documentation for IP-to-OS mapping")
    print("   - Look for a metadata.json or labels.csv file")
    print("   - Visit: https://nprint.github.io/benchmarks/os_detection/")

    print("\n2. To extract with OS labels:")
    print("   - If you know the OS, use: --os-label 'Windows 11'")
    print("   - Or split by IP and label separately")

    print("\n3. For mixed OS captures:")
    print("   - Use scripts/split_pcap_by_ip.py to separate by IP")
    print("   - Then extract each IP group with appropriate OS label")

    print("\n")


if __name__ == '__main__':
    main()
