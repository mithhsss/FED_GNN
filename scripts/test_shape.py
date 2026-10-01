import urllib.parse, urllib.request
from PIL import Image
import io

dot = """
digraph G {
    graph [rankdir=TB, bgcolor="white", nodesep=0.3, ranksep=0.4, pad="0.2"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9.5];
    edge [fontname="Helvetica", fontsize=8.5];

    subgraph cluster_1 {
        label="Stage 1: Ingestion & Zero-Leakage Split";
        flows -> split;
    }
    subgraph cluster_2 {
        label="Stage 2: Multi-Topology Graph Construction";
        {rank=same; t; c; b;}
    }
    split -> t; split -> c; split -> b;
    subgraph cluster_3 {
        label="Stage 3: Federated Edge Clients (K=5)";
        {rank=same; gt; gc; gb;}
        {rank=same; leiden; pool; loss;}
        leiden -> pool;
    }
    t -> gt; c -> gc; b -> gb;
    gt -> leiden; gc -> leiden; gb -> leiden;
    subgraph cluster_4 {
        label="Stage 4: Server Coordination & Overlay";
        {rank=same; fed; sage;}
    }
    gt -> fed; gc -> fed; gb -> fed;
    pool -> sage;
    subgraph cluster_5 {
        label="Stage 5: Ensemble Decision Arbitration";
        meta -> rf -> out;
    }
    gt -> meta; gc -> meta; gb -> meta; sage -> meta;
    split -> meta [label="Raw Flow Skip", style=dashed, constraint=false];
}
"""
encoded = urllib.parse.quote(dot)
url = f'https://quickchart.io/graphviz?graph={encoded}&format=png'
data = urllib.request.urlopen(url).read()
img = Image.open(io.BytesIO(data))
print('Size with constraint=false:', img.size)
