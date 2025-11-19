# Quick diagnostic script: check_nprint_packets.py
from scapy.all import rdpcap, IP, TCP
from collections import Counter

pcap_file = 'data/raw/nprint/os-100-packet.pcapng'

print(f"Analyzing {pcap_file}...")
packets = rdpcap(pcap_file)

total = len(packets)
tcp_packets = [p for p in packets if p.haslayer(TCP)]
tcp_count = len(tcp_packets)

# Analyze TCP flags
flag_types = Counter()
syn_packets = []

for pkt in tcp_packets:
    tcp = pkt[TCP]
    flags = tcp.flags
    
    if flags & 0x02:  # SYN
        if flags & 0x10:  # SYN-ACK
            flag_types['SYN-ACK'] += 1
        else:
            flag_types['SYN'] += 1
            syn_packets.append(pkt)
    elif flags & 0x10:  # ACK (without SYN)
        if flags & 0x08:  # PSH-ACK
            flag_types['PSH-ACK'] += 1
        else:
            flag_types['ACK'] += 1
    elif flags & 0x01:  # FIN
        flag_types['FIN'] += 1
    elif flags & 0x04:  # RST
        flag_types['RST'] += 1

print(f"\nTotal packets: {total}")
print(f"TCP packets: {tcp_count} ({100*tcp_count/total:.1f}%)")
print(f"\nTCP Flag Distribution:")
for flag, count in sorted(flag_types.items(), key=lambda x: -x[1]):
    pct = 100 * count / tcp_count
    print(f"  {flag:12s}: {count:6,} ({pct:5.1f}%)")

print(f"\n{'='*50}")
print(f"SYN packets with TCP options: {len(syn_packets)}")

# Check TCP options in SYN packets
if syn_packets:
    print(f"\nSample SYN packet analysis:")
    pkt = syn_packets[0]
    tcp = pkt[TCP]
    print(f"  TCP options: {tcp.options}")
    print(f"  Window size: {tcp.window}")
    print(f"  TTL: {pkt[IP].ttl if pkt.haslayer(IP) else 'N/A'}")