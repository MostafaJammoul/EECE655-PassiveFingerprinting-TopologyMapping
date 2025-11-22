#!/usr/bin/env python3
"""
Extract Flow Features from nprint PCAP for OS Family Classification

Input:  data/raw/nprint/os-100-packet.pcapng
Output: data/processed/nprint_flows.csv

Extracts the same 25 features as masaryk_preprocess.py from PCAP flows.
Filtering: Only TCP flows with SYN flag present (matches Masaryk approach)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from datetime import datetime

try:
    from scapy.all import rdpcap, TCP, IP, IPv6, Raw
    # Try multiple import paths for TLS (depends on scapy version)
    try:
        from scapy.layers.tls.record import TLS
        from scapy.layers.tls.handshake import TLSClientHello
        from scapy.layers.tls.extensions import TLS_Ext_SupportedGroups, TLS_Ext_ServerName
    except ImportError:
        try:
            # Older scapy versions
            from scapy.layers.ssl_tls import TLS, TLSClientHello
            TLS_Ext_SupportedGroups = None
            TLS_Ext_ServerName = None
        except ImportError:
            # Fallback - TLS parsing will be disabled
            TLS = None
            TLSClientHello = None
            TLS_Ext_SupportedGroups = None
            TLS_Ext_ServerName = None
            print("WARNING: TLS layer not available in this scapy version. TLS features will be empty.")
except ImportError:
    print("ERROR: scapy not installed")
    print("\nInstall with:")
    print("  pip install scapy")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_initial_ttl(ttl):
    """Estimate original TTL value based on observed TTL"""
    if ttl is None:
        return None
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255


def tcp_flags_to_string(flags_int):
    """
    Convert TCP flags integer to Masaryk string format
    Format: C E U A P R S F
    """
    if flags_int is None:
        return None

    result = ""
    result += "C" if (flags_int & 0x80) else "-"  # CWR
    result += "E" if (flags_int & 0x40) else "-"  # ECE
    result += "U" if (flags_int & 0x20) else "-"  # URG
    result += "A" if (flags_int & 0x10) else "-"  # ACK
    result += "P" if (flags_int & 0x08) else "-"  # PSH
    result += "R" if (flags_int & 0x04) else "-"  # RST
    result += "S" if (flags_int & 0x02) else "-"  # SYN
    result += "F" if (flags_int & 0x01) else "-"  # FIN
    return result


def extract_tcp_options(tcp_layer):
    """Extract TCP options from TCP layer"""
    options = {
        'window_scale': None,
        'sack_permitted': None,
        'mss': None,
        'nop_count': 0
    }

    if not hasattr(tcp_layer, 'options') or not tcp_layer.options:
        return options

    for opt in tcp_layer.options:
        if isinstance(opt, tuple):
            opt_name = opt[0]
            if opt_name == 'WScale':
                options['window_scale'] = opt[1]
            elif opt_name == 'SAckOK':
                options['sack_permitted'] = 1
            elif opt_name == 'MSS':
                options['mss'] = opt[1]
            elif opt_name == 'NOP':
                options['nop_count'] += 1

    return options


def extract_tls_features(packet):
    """Extract TLS features from ClientHello packet"""
    tls_features = {
        'ja3_fingerprint': None,
        'cipher_suites': None,
        'extension_types': None,
        'elliptic_curves': None,
        'client_version': None,
        'handshake_type': None,
        'client_key_length': None
    }

    # Skip TLS parsing if TLS layer not available
    if TLS is None or TLSClientHello is None:
        return tls_features

    try:
        if packet.haslayer(TLS):
            tls_layer = packet[TLS]

            # Check for ClientHello
            if hasattr(tls_layer, 'msg') and tls_layer.msg:
                for msg in tls_layer.msg:
                    if isinstance(msg, TLSClientHello):
                        # TLS version
                        if hasattr(msg, 'version'):
                            tls_features['client_version'] = str(msg.version)

                        # Handshake type
                        tls_features['handshake_type'] = str(msg.msgtype) if hasattr(msg, 'msgtype') else None

                        # Cipher suites
                        if hasattr(msg, 'ciphers') and msg.ciphers:
                            cipher_hex = ''.join([f'{c:04X}' for c in msg.ciphers])
                            tls_features['cipher_suites'] = cipher_hex

                        # Extensions
                        if hasattr(msg, 'ext') and msg.ext:
                            ext_types = []
                            curves = []
                            for ext in msg.ext:
                                # Extension type
                                if hasattr(ext, 'type'):
                                    ext_types.append(f'{ext.type:04X}')

                                # Elliptic curves
                                if TLS_Ext_SupportedGroups and isinstance(ext, TLS_Ext_SupportedGroups):
                                    if hasattr(ext, 'groups'):
                                        curves = [f'{g:04X}' for g in ext.groups]

                            if ext_types:
                                tls_features['extension_types'] = ''.join(ext_types)
                            if curves:
                                tls_features['elliptic_curves'] = ''.join(curves)

                        # Key length (from cipher suite - simplified)
                        if tls_features['cipher_suites']:
                            # Common mapping: some ciphers use specific key lengths
                            # This is a simplified approximation
                            tls_features['client_key_length'] = 528  # Common default

                        # JA3 fingerprint (simplified MD5 of TLS parameters)
                        # Full JA3 requires: version,ciphers,extensions,curves,formats
                        if tls_features['client_version'] and tls_features['cipher_suites']:
                            import hashlib
                            ja3_string = f"{tls_features['client_version']}," \
                                       f"{tls_features['cipher_suites']}," \
                                       f"{tls_features['extension_types'] or ''}," \
                                       f"{tls_features['elliptic_curves'] or ''}"
                            tls_features['ja3_fingerprint'] = hashlib.md5(ja3_string.encode()).hexdigest().upper()

                        break
    except Exception as e:
        # TLS parsing can be fragile, continue on error
        pass

    return tls_features


# Windows Expert Model IP-to-Label Mapping
# Maps specific IP addresses to Windows version labels
WINDOWS_IP_MAPPING = {
    '192.168.10.9': 'Windows 7',      # Win 7 Pro, 64B
    '192.168.10.5': 'Windows 8',      # Win 8.1, 64B (treated as Windows 8)
    '192.168.10.8': 'Windows Vista',  # Win Vista, 64B
    '192.168.10.14': 'Windows 10',    # Win 10, pro 32B
    '192.168.10.15': 'Windows 10',    # Win 10, 64B
}

# Flow timeout settings (matches NetFlow/IPFIX behavior)
FLOW_IDLE_TIMEOUT = 15.0   # seconds - flow ends if no packets for 15s
FLOW_ACTIVE_TIMEOUT = 60.0  # seconds - flow ends after 60s from start


def apply_flow_timeouts(flow_packets, idle_timeout=FLOW_IDLE_TIMEOUT,
                       active_timeout=FLOW_ACTIVE_TIMEOUT):
    """
    Apply idle and active timeouts to flow packets

    Mimics NetFlow/IPFIX flow export behavior:
    - IDLE timeout: Flow ends if gap between packets > idle_timeout
    - ACTIVE timeout: Flow ends after active_timeout from first packet

    Args:
        flow_packets: List of packet dicts with 'timestamp' key (sorted by time)
        idle_timeout: Maximum idle time between packets (seconds)
        active_timeout: Maximum total flow duration (seconds)

    Returns:
        Filtered list of packets within timeout constraints
    """
    if not flow_packets or len(flow_packets) == 0:
        return flow_packets

    # Get first packet timestamp (flow start)
    flow_start_time = flow_packets[0]['timestamp']

    # Filter packets based on timeouts
    valid_packets = []
    prev_packet_time = flow_start_time

    for pkt_info in flow_packets:
        current_time = pkt_info['timestamp']

        # Check active timeout (time since flow start)
        if current_time - flow_start_time > active_timeout:
            break  # Flow exceeded active timeout

        # Check idle timeout (time since last packet)
        if current_time - prev_packet_time > idle_timeout:
            break  # Flow exceeded idle timeout

        valid_packets.append(pkt_info)
        prev_packet_time = current_time

    return valid_packets


def get_os_label_from_ip(ip_address, ip_mapping=None):
    """
    Get OS label from IP address using IP mapping

    Args:
        ip_address: IP address string
        ip_mapping: Dict mapping IPs to OS labels

    Returns:
        OS label string or None if not in mapping
    """
    if ip_mapping is None:
        ip_mapping = WINDOWS_IP_MAPPING

    return ip_mapping.get(ip_address)


def extract_os_label_from_pcap(pcap_path):
    """
    Extract OS label from PCAP filename or metadata

    For nprint dataset, the filename might contain OS info.
    Example: "windows-10.pcap" -> "Windows"
    """
    filename = Path(pcap_path).stem.lower()

    if 'windows' in filename or 'win' in filename:
        return 'Windows'
    elif 'linux' in filename or 'ubuntu' in filename or 'debian' in filename:
        return 'Linux'
    elif 'macos' in filename or 'darwin' in filename or 'osx' in filename:
        return 'macOS'
    elif 'android' in filename:
        return 'Android'
    elif 'ios' in filename or 'iphone' in filename:
        return 'iOS'
    else:
        return 'Unknown'


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def process_pcap(pcap_path, filter_mode='syn_required', ip_mapping=None, verbose=True):
    """
    Process PCAP file and extract flow-level features

    Applies NetFlow/IPFIX-style flow timeouts:
    - IDLE timeout (15s): Flow ends if no packets for 15 seconds
    - ACTIVE timeout (60s): Flow ends after 60 seconds from start

    Args:
        pcap_path: Path to PCAP file
        filter_mode: 'syn_required' (default), 'tls_only', or 'all'
        ip_mapping: Dict mapping IPs to OS labels (filters flows to these IPs only)
        verbose: Print progress

    Returns:
        DataFrame with extracted features
    """

    if verbose:
        print("="*70)
        print("NPRINT PCAP FLOW EXTRACTION")
        print("="*70)
        print(f"\nInput:  {pcap_path}")
        print(f"Filter: {filter_mode}")
        print(f"Flow Timeouts: IDLE={FLOW_IDLE_TIMEOUT}s, ACTIVE={FLOW_ACTIVE_TIMEOUT}s")
        if ip_mapping:
            print(f"IP Filter: {len(ip_mapping)} specific SOURCE IPs (flows initiated by)")
            for ip, label in ip_mapping.items():
                print(f"  {ip} -> {label}")

    # Load PCAP
    if verbose:
        print(f"\n[1/3] Loading PCAP file...")

    try:
        packets = rdpcap(str(pcap_path))
        if verbose:
            print(f"  Loaded {len(packets):,} packets")
    except Exception as e:
        print(f"ERROR: Failed to load PCAP: {e}")
        return None

    # Group packets into flows (5-tuple)
    if verbose:
        print(f"\n[2/3] Grouping packets into flows...")

    flows = defaultdict(list)

    for pkt in tqdm(packets, desc="Processing packets", disable=not verbose):
        if pkt.haslayer(TCP) and (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
            # Get IP layer
            ip_layer = pkt[IP] if pkt.haslayer(IP) else pkt[IPv6]
            tcp_layer = pkt[TCP]

            # Get actual packet IPs/ports
            pkt_src_ip = ip_layer.src
            pkt_dst_ip = ip_layer.dst
            pkt_src_port = tcp_layer.sport
            pkt_dst_port = tcp_layer.dport

            # If filtering by IP mapping, only include packets involving mapped IPs
            if ip_mapping:
                # Check if either IP is in the mapping
                if pkt_src_ip in ip_mapping or pkt_dst_ip in ip_mapping:
                    # Create normalized flow key (always put mapped IP first)
                    if pkt_src_ip in ip_mapping:
                        flow_key = (pkt_src_ip, pkt_dst_ip, pkt_src_port, pkt_dst_port, 6)
                    else:
                        # Reverse so mapped IP is first
                        flow_key = (pkt_dst_ip, pkt_src_ip, pkt_dst_port, pkt_src_port, 6)
                else:
                    # Neither IP is in mapping - skip
                    continue
            else:
                # No IP mapping - use standard normalization (alphabetical)
                if (pkt_src_ip, pkt_src_port) < (pkt_dst_ip, pkt_dst_port):
                    flow_key = (pkt_src_ip, pkt_dst_ip, pkt_src_port, pkt_dst_port, 6)
                else:
                    flow_key = (pkt_dst_ip, pkt_src_ip, pkt_dst_port, pkt_src_port, 6)

            # Store packet with its ACTUAL src/dst (not normalized)
            # We'll determine direction later based on SYN packet
            flows[flow_key].append({
                'packet': pkt,
                'pkt_src_ip': pkt_src_ip,
                'pkt_dst_ip': pkt_dst_ip,
                'pkt_src_port': pkt_src_port,
                'pkt_dst_port': pkt_dst_port,
                'timestamp': float(pkt.time) if hasattr(pkt, 'time') else 0
            })

    if verbose:
        print(f"  Found {len(flows):,} unique flows")

        # DEBUG: Check first few flows for bidirectional packets
        if ip_mapping:
            print(f"\n  DEBUG: Checking first 3 flows for bidirectional packets...")
            for i, (flow_key, pkts) in enumerate(list(flows.items())[:3]):
                src_ip, dst_ip, src_port, dst_port, proto = flow_key
                print(f"    Flow {i+1}: {src_ip}:{src_port} -> {dst_ip}:{dst_port}")
                print(f"      Total packets: {len(pkts)}")

                # Count packets by actual direction (before SYN-based assignment)
                client_to_server = sum(1 for p in pkts if p['pkt_src_ip'] == src_ip and p['pkt_src_port'] == src_port)
                server_to_client = sum(1 for p in pkts if p['pkt_src_ip'] == dst_ip and p['pkt_src_port'] == dst_port)
                print(f"      Client->Server: {client_to_server}, Server->Client: {server_to_client}")

    # Extract features from flows
    if verbose:
        print(f"\n[3/3] Extracting features from flows...")

    records = []
    filtered_no_syn = 0
    filtered_no_tls = 0
    filtered_no_ip_match = 0

    default_os_label = extract_os_label_from_pcap(pcap_path)

    for flow_key, flow_packets in tqdm(flows.items(), desc="Extracting features", disable=not verbose):
        src_ip, dst_ip, src_port, dst_port, proto = flow_key

        # Get OS label from IP mapping or filename
        if ip_mapping:
            # When using IP mapping, src_ip is always the Windows machine IP
            # (due to flow key construction in packet processing above)
            os_label = get_os_label_from_ip(src_ip, ip_mapping)
            if not os_label:
                # This shouldn't happen anymore since we filter at packet level
                filtered_no_ip_match += 1
                continue
        else:
            os_label = default_os_label

        # Sort packets by timestamp
        flow_packets = sorted(flow_packets, key=lambda x: x['timestamp'])

        # Apply flow timeouts (idle and active)
        # This mimics NetFlow/IPFIX behavior and matches Masaryk dataset
        flow_packets = apply_flow_timeouts(flow_packets)

        # Find SYN packet - ONLY pure SYN (client initiator), NOT SYN-ACK
        syn_packet = None
        syn_packet_info = None
        has_pure_syn = False

        for pkt_info in flow_packets:
            pkt = pkt_info['packet']
            if pkt.haslayer(TCP):
                tcp_layer = pkt[TCP]
                flags = tcp_layer.flags
                # Look for pure SYN (SYN set, ACK not set) - this is the connection initiator
                if (flags & 0x02) and not (flags & 0x10):  # SYN=1, ACK=0
                    syn_packet = pkt
                    syn_packet_info = pkt_info
                    has_pure_syn = True
                    break

        # Determine forward direction based on PURE SYN packet (client initiator)
        # Forward = direction of pure SYN packet (client → server)
        # Backward = opposite direction (server → client)
        #
        # IMPORTANT: We do NOT use SYN-ACK to determine direction!
        # SYN-ACK is a server response (backward direction), not client initiator
        if has_pure_syn and syn_packet_info:
            syn_src_ip = syn_packet_info['pkt_src_ip']
            syn_src_port = syn_packet_info['pkt_src_port']
            syn_dst_ip = syn_packet_info['pkt_dst_ip']
            syn_dst_port = syn_packet_info['pkt_dst_port']

            # CRITICAL FILTER: When using IP mapping, ONLY keep flows where
            # the Windows machine (mapped IP) sent the SYN packet (client role)
            # Skip flows where Windows machine is the SYN receiver (server role)
            if ip_mapping:
                if syn_src_ip not in ip_mapping:
                    # SYN was sent by external host TO Windows machine
                    # This means Windows is acting as SERVER, not client
                    # Skip this flow per user's requirement
                    filtered_no_ip_match += 1
                    continue

            # Assign direction to all packets based on SYN direction
            for pkt_info in flow_packets:
                if (pkt_info['pkt_src_ip'] == syn_src_ip and
                    pkt_info['pkt_src_port'] == syn_src_port and
                    pkt_info['pkt_dst_ip'] == syn_dst_ip and
                    pkt_info['pkt_dst_port'] == syn_dst_port):
                    pkt_info['direction'] = 'forward'
                else:
                    pkt_info['direction'] = 'backward'
        else:
            # No pure SYN found - when using IP mapping, we CANNOT determine
            # if Windows machine was the client or server without a SYN packet
            # User requirement: ONLY keep flows initiated BY Windows machines
            # So skip flows without pure SYN when filtering by IP
            if ip_mapping:
                # No pure SYN found AND we're filtering by IP mapping
                # Cannot confirm Windows initiated this flow - skip it
                filtered_no_syn += 1
                continue

            # No IP mapping - use flow key to determine direction
            # Flow key has client IP first (from flow aggregation logic)
            # Packets matching flow key order are forward, opposite are backward
            for pkt_info in flow_packets:
                if (pkt_info['pkt_src_ip'] == src_ip and
                    pkt_info['pkt_src_port'] == src_port and
                    pkt_info['pkt_dst_ip'] == dst_ip and
                    pkt_info['pkt_dst_port'] == dst_port):
                    pkt_info['direction'] = 'forward'
                else:
                    pkt_info['direction'] = 'backward'

        # Check if we have any SYN packet (including SYN-ACK) for filtering
        has_syn = has_pure_syn
        if not has_syn:
            for pkt_info in flow_packets:
                pkt = pkt_info['packet']
                if pkt.haslayer(TCP):
                    tcp_layer = pkt[TCP]
                    flags = tcp_layer.flags
                    if flags & 0x02:  # Any SYN (for filtering purposes only)
                        syn_packet = pkt  # Keep for TCP feature extraction
                        has_syn = True
                        break

        # Apply filtering
        if filter_mode == 'syn_required' and not has_syn:
            filtered_no_syn += 1
            continue

        if filter_mode == 'tls_only':
            has_tls = any(p['packet'].haslayer(TLS) for p in flow_packets)
            if not has_tls:
                filtered_no_tls += 1
                continue

        # Extract features
        feature_dict = {}

        # Get first packet for basic info
        first_pkt = flow_packets[0]['packet']
        ip_layer = first_pkt[IP] if first_pkt.haslayer(IP) else first_pkt[IPv6]

        # Aggregate ALL TCP flags seen across the entire flow (matches Masaryk)
        # This gives us flags like ---AP-SF (ACK, PSH, SYN, FIN) instead of just ------S-
        all_tcp_flags = 0
        for pkt_info in flow_packets:
            pkt = pkt_info['packet']
            if pkt.haslayer(TCP):
                all_tcp_flags |= pkt[TCP].flags

        # Check if we saw a SYN-ACK packet (server response)
        # SYN-ACK = SYN flag set AND ACK flag set
        has_syn_ack = False
        for pkt_info in flow_packets:
            pkt = pkt_info['packet']
            if pkt.haslayer(TCP):
                flags = pkt[TCP].flags
                if (flags & 0x02) and (flags & 0x10):  # SYN=1 AND ACK=1
                    has_syn_ack = True
                    break

        # TCP features from SYN packet (if available)
        if syn_packet:
            tcp_syn = syn_packet[TCP]
            ip_syn = syn_packet[IP] if syn_packet.haslayer(IP) else syn_packet[IPv6]

            feature_dict['tcp_syn_size'] = len(tcp_syn)
            feature_dict['tcp_win_size'] = tcp_syn.window
            feature_dict['tcp_syn_ttl'] = ip_syn.ttl if hasattr(ip_syn, 'ttl') else ip_syn.hlim
            feature_dict['tcp_flags_a'] = tcp_flags_to_string(all_tcp_flags)  # Use aggregated flags
            feature_dict['syn_ack_flag'] = 1 if has_syn_ack else 0  # Only set if we saw SYN-ACK

            # TCP options
            tcp_opts = extract_tcp_options(tcp_syn)
            feature_dict['tcp_option_window_scale_forward'] = tcp_opts['window_scale']
            feature_dict['tcp_option_selective_ack_permitted_forward'] = tcp_opts['sack_permitted']
            feature_dict['tcp_option_maximum_segment_size_forward'] = tcp_opts['mss']
            feature_dict['tcp_option_no_operation_forward'] = tcp_opts['nop_count'] if tcp_opts['nop_count'] > 0 else None
        else:
            # No SYN packet, use first packet
            feature_dict['tcp_syn_size'] = None
            feature_dict['tcp_win_size'] = None
            feature_dict['tcp_syn_ttl'] = None
            feature_dict['tcp_flags_a'] = tcp_flags_to_string(all_tcp_flags) if all_tcp_flags > 0 else None
            feature_dict['syn_ack_flag'] = 1 if has_syn_ack else 0  # Can have SYN-ACK without pure SYN
            feature_dict['tcp_option_window_scale_forward'] = None
            feature_dict['tcp_option_selective_ack_permitted_forward'] = None
            feature_dict['tcp_option_maximum_segment_size_forward'] = None
            feature_dict['tcp_option_no_operation_forward'] = None

        # IP features
        feature_dict['l3_proto'] = 4 if first_pkt.haslayer(IP) else 6
        feature_dict['ip_tos'] = ip_layer.tos if hasattr(ip_layer, 'tos') else 0

        # Calculate max TTL forward
        max_ttl_fwd = 0
        for pkt_info in flow_packets:
            if pkt_info['direction'] == 'forward':
                pkt = pkt_info['packet']
                ip = pkt[IP] if pkt.haslayer(IP) else pkt[IPv6]
                ttl = ip.ttl if hasattr(ip, 'ttl') else ip.hlim
                max_ttl_fwd = max(max_ttl_fwd, ttl)

        feature_dict['maximum_ttl_forward'] = max_ttl_fwd if max_ttl_fwd > 0 else None

        # DF flag (IPv4 only)
        if first_pkt.haslayer(IP):
            feature_dict['ipv4_dont_fragment_forward'] = 1 if (ip_layer.flags & 0x2) else 0
        else:
            feature_dict['ipv4_dont_fragment_forward'] = None

        # Flow metadata
        feature_dict['src_port'] = src_port

        # Packet counts
        fwd_count = sum(1 for p in flow_packets if p.get('direction') == 'forward')
        bwd_count = sum(1 for p in flow_packets if p.get('direction') == 'backward')

        # DEBUG: Check if any packet is missing direction
        missing_direction = sum(1 for p in flow_packets if 'direction' not in p)
        if missing_direction > 0 and verbose:
            print(f"  WARNING: {missing_direction} packets missing direction in flow {src_ip}:{src_port} -> {dst_ip}:{dst_port}")

        # DEBUG: Print first flow with backward packets
        if bwd_count > 0 and verbose:
            print(f"\n  DEBUG: Flow with backward packets found!")
            print(f"    Flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port}")
            print(f"    Forward: {fwd_count}, Backward: {bwd_count}")
            print(f"    Sample backward packet: {[p for p in flow_packets if p.get('direction') == 'backward'][0]['pkt_src_ip']}:{[p for p in flow_packets if p.get('direction') == 'backward'][0]['pkt_src_port']}")

        feature_dict['packet_total_count_forward'] = fwd_count
        feature_dict['packet_total_count_backward'] = bwd_count

        # Total bytes
        total_bytes = sum(len(p['packet']) for p in flow_packets)
        feature_dict['total_bytes'] = float(total_bytes)

        # TLS features (search for ClientHello)
        tls_features = {'ja3_fingerprint': None, 'cipher_suites': None,
                       'extension_types': None, 'elliptic_curves': None,
                       'client_version': None, 'handshake_type': None,
                       'client_key_length': None}

        for pkt_info in flow_packets:
            pkt = pkt_info['packet']
            if pkt.haslayer(TLS):
                tls_features = extract_tls_features(pkt)
                if tls_features['ja3_fingerprint']:  # Found ClientHello
                    break

        feature_dict.update({
            'tls_ja3_fingerprint': tls_features['ja3_fingerprint'],
            'tls_cipher_suites': tls_features['cipher_suites'],
            'tls_extension_types': tls_features['extension_types'],
            'tls_elliptic_curves': tls_features['elliptic_curves'],
            'tls_client_version': tls_features['client_version'],
            'tls_handshake_type': tls_features['handshake_type'],
            'tls_client_key_length': tls_features['client_key_length']
        })

        # Derived features
        feature_dict['initial_ttl'] = calculate_initial_ttl(feature_dict.get('tcp_syn_ttl'))

        # OS family (from filename or provided)
        feature_dict['os_family'] = os_label

        records.append(feature_dict)

    if verbose:
        print(f"\n  Processed {len(flows):,} flows")
        if ip_mapping:
            print(f"  Filtered (IP not in mapping): {filtered_no_ip_match:,}")
        if filter_mode == 'syn_required':
            print(f"  Filtered (no SYN): {filtered_no_syn:,}")
        if filter_mode == 'tls_only':
            print(f"  Filtered (no TLS): {filtered_no_tls:,}")
        print(f"  Extracted features: {len(records):,} flows")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Ensure column order matches Masaryk
    column_order = [
        'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a', 'syn_ack_flag',
        'tcp_option_window_scale_forward', 'tcp_option_selective_ack_permitted_forward',
        'tcp_option_maximum_segment_size_forward', 'tcp_option_no_operation_forward',
        'l3_proto', 'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward',
        'src_port', 'packet_total_count_forward', 'packet_total_count_backward', 'total_bytes',
        'tls_ja3_fingerprint', 'tls_cipher_suites', 'tls_extension_types',
        'tls_elliptic_curves', 'tls_client_version', 'tls_handshake_type', 'tls_client_key_length',
        'initial_ttl', 'os_family'
    ]

    df = df[column_order]

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract flow features from nprint PCAP file'
    )

    parser.add_argument(
        '--pcap',
        type=str,
        default='data/raw/nprint/os-100-packet.pcapng',
        help='Path to PCAP file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/nprint_flows.csv',
        help='Output CSV file'
    )

    parser.add_argument(
        '--filter',
        type=str,
        default='syn_required',
        choices=['syn_required', 'tls_only', 'all'],
        help='Filtering mode: syn_required (default), tls_only, or all'
    )

    parser.add_argument(
        '--os-label',
        type=str,
        default=None,
        help='OS family label (if not in filename). Default: auto-detect from filename'
    )

    parser.add_argument(
        '--use-windows-ips',
        action='store_true',
        help='Filter to flows INITIATED by Windows expert model IPs (192.168.10.5/8/9/14/15)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Determine IP mapping
    ip_mapping = None
    if args.use_windows_ips:
        ip_mapping = WINDOWS_IP_MAPPING
        if verbose:
            print(f"\n🎯 Using Windows Expert Model IP filtering (SOURCE IPs only)")
            print(f"   Filtering to flows INITIATED by {len(ip_mapping)} specific IPs:")
            for ip, label in ip_mapping.items():
                print(f"     {ip} -> {label}")

    # Process PCAP
    df = process_pcap(args.pcap, filter_mode=args.filter, ip_mapping=ip_mapping, verbose=verbose)

    if df is None or len(df) == 0:
        print("\nERROR: No flows extracted!")
        sys.exit(1)

    # Override OS label if provided (only when not using IP mapping)
    if args.os_label and not ip_mapping:
        df['os_family'] = args.os_label

    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    if verbose:
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"\nOutput: {args.output}")
        print(f"Flows extracted: {len(df):,}")
        print(f"\nOS Family distribution:")
        print(df['os_family'].value_counts())

        print(f"\nFeature completeness:")
        print(f"  TCP features: {df['tcp_syn_size'].notna().sum() / len(df) * 100:.1f}%")
        print(f"  TLS features: {df['tls_ja3_fingerprint'].notna().sum() / len(df) * 100:.1f}%")

        print(f"\nSample row:")
        print(df.head(1).to_string())


if __name__ == '__main__':
    main()
