import time
import os
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import community as community_louvain

def main():
    print("=" * 80)
    print("FULL GRAPH COMMUNITY AUDIT: LOUVAIN vs. LEIDEN (1,501 NODES, 1.15M FLOWS)")
    print("=" * 80)

    csv_path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
    df = pd.read_csv(csv_path, usecols=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'])
    
    all_ips = list(set(df['IPV4_SRC_ADDR']).union(set(df['IPV4_DST_ADDR'])))
    ip2idx = {ip: i for i, ip in enumerate(all_ips)}
    src_idx = [ip2idx[ip] for ip in df['IPV4_SRC_ADDR']]
    dst_idx = [ip2idx[ip] for ip in df['IPV4_DST_ADDR']]

    # 1. Build NetworkX graph
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(len(all_ips)))
    
    # Fast edge aggregation
    edge_counts = {}
    for s, d in zip(src_idx, dst_idx):
        if s > d:
            s, d = d, s
        edge_counts[(s, d)] = edge_counts.get((s, d), 0) + 1

    for (s, d), w in edge_counts.items():
        G_nx.add_edge(s, d, weight=w)

    print(f"Graph Topology: {G_nx.number_of_nodes():,} IP nodes, {G_nx.number_of_edges():,} communication edges")

    # 2. Run Louvain
    t0 = time.time()
    louvain_part = community_louvain.best_partition(G_nx, weight='weight', random_state=42)
    t_louvain = time.time() - t0
    num_louvain_comms = len(set(louvain_part.values()))

    # 3. Build iGraph & Run Leiden
    t0 = time.time()
    edges = list(G_nx.edges())
    weights = [float(G_nx[u][v].get('weight', 1.0)) for u, v in edges]
    g_ig = ig.Graph(n=len(all_ips), edges=edges, directed=False)
    g_ig.es['weight'] = weights

    # Use ModularityVertexPartition (Traag et al. 2019)
    leiden_partition_obj = leidenalg.find_partition(
        g_ig,
        leidenalg.ModularityVertexPartition,
        weights='weight',
        seed=42
    )
    t_leiden = time.time() - t0
    leiden_part = {i: c for i, c in enumerate(leiden_partition_obj.membership)}
    num_leiden_comms = len(set(leiden_part.values()))

    # 4. Connectedness and Internal Density Analysis
    def analyze_partition(G, part_dict, name):
        comms = {}
        for node, c in part_dict.items():
            comms.setdefault(c, []).append(node)
        
        disconnected_count = 0
        internal_densities = []
        
        for c, nodes in comms.items():
            if len(nodes) <= 1:
                internal_densities.append(1.0)
                continue
            subg = G.subgraph(nodes)
            if not nx.is_connected(subg):
                disconnected_count += 1
            n_nodes = len(nodes)
            n_edges = subg.number_of_edges()
            max_edges = n_nodes * (n_nodes - 1) / 2.0
            density = n_edges / max_edges if max_edges > 0 else 0.0
            internal_densities.append(density)

        dis_pct = (disconnected_count / len(comms)) * 100.0
        avg_density = np.mean(internal_densities)
        mod = community_louvain.modularity(part_dict, G, weight='weight')
        return {
            'name': name,
            'num_comms': len(comms),
            'disconnected_count': disconnected_count,
            'disconnected_pct': dis_pct,
            'avg_density': avg_density,
            'modularity': mod
        }

    louvain_stats = analyze_partition(G_nx, louvain_part, "Louvain")
    leiden_stats = analyze_partition(G_nx, leiden_part, "Leiden")

    print("\n" + "=" * 80)
    print(f"{'Metric':<35} {'Louvain':>15} {'Leiden':>15} {'Diff / Status':>15}")
    print("=" * 80)
    print(f"{'Execution Time':<35} {t_louvain:>14.4f}s {t_leiden:>14.4f}s {t_leiden - t_louvain:>+14.4f}s")
    print(f"{'Total Communities':<35} {louvain_stats['num_comms']:>15} {leiden_stats['num_comms']:>15} {leiden_stats['num_comms'] - louvain_stats['num_comms']:>+15}")
    print(f"{'Modularity Score':<35} {louvain_stats['modularity']:>15.4f} {leiden_stats['modularity']:>15.4f} {leiden_stats['modularity'] - louvain_stats['modularity']:>+15.4f}")
    print(f"{'Internally Disconnected Comms':<35} {louvain_stats['disconnected_count']:>15} {leiden_stats['disconnected_count']:>15} {leiden_stats['disconnected_count'] - louvain_stats['disconnected_count']:>+15}")
    print(f"{'Disconnected Comms %':<35} {louvain_stats['disconnected_pct']:>14.2f}% {leiden_stats['disconnected_pct']:>14.2f}% {leiden_stats['disconnected_pct'] - louvain_stats['disconnected_pct']:>+14.2f}%")
    print(f"{'Avg Internal Community Density':<35} {louvain_stats['avg_density']:>15.4f} {leiden_stats['avg_density']:>15.4f} {leiden_stats['avg_density'] - louvain_stats['avg_density']:>+15.4f}")
    print("=" * 80)
    
if __name__ == '__main__':
    main()
