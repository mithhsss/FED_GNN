import urllib.request, urllib.parse
from PIL import Image
import io, os

dot_code = """digraph G {
    graph [rankdir=TB, bgcolor="white", fontname="Helvetica", fontsize=11, compound=true, nodesep=0.35, ranksep=0.45, pad="0.2"];
    node [fontname="Helvetica", fontsize=9.5, shape=box, style="filled,rounded", penwidth=1.4, margin="0.15,0.10"];
    edge [fontname="Helvetica", fontsize=8.5, color="#1f4e79", penwidth=1.2, fontcolor="#0b2238"];

    // =========================================================================
    // STAGE 1: INGESTION & ZERO-LEAKAGE SPLIT
    // =========================================================================
    subgraph cluster_0 {
        label = "Stage 1: Ingestion & Evaluation Control";
        color = "#0d6efd";
        style = "dashed,rounded";
        bgcolor = "#f8faff";
        fontcolor = "#084298";
        fontsize = 11;

        flows [label="NF-ToN-IoT NetFlow Stream (1,377,837 Flows, 24 Attributes)", fillcolor="#e7f1ff", color="#0d6efd", shape=ellipse, style="filled,bold"];
        sig_split [label="8-Tuple Signature Grouping (StratifiedGroupKFold)\\n[0.00% Flow/Socket Evaluation Leakage]", fillcolor="#cfe2ff", color="#0d6efd"];
        
        flows -> sig_split;
    }

    // =========================================================================
    // STAGE 2: MULTI-TOPOLOGY GRAPH DECOMPOSITION
    // =========================================================================
    subgraph cluster_1 {
        label = "Stage 2: Multi-Topology Feature Routing & Subgraph Construction";
        color = "#198754";
        style = "dashed,rounded";
        bgcolor = "#f4faf6";
        fontcolor = "#0f5132";
        fontsize = 11;

        { rank=same; feat_temp; feat_cont; feat_behav; }

        feat_temp [label="Temporal Attributes\\n(Duration, Rates, Ratios)", fillcolor="#d1e7dd", color="#198754"];
        feat_cont [label="Content Attributes\\n(Byte Volumes, TCP Flags)", fillcolor="#d1e7dd", color="#198754"];
        feat_behav [label="Behavioral Attributes\\n(Priv Ports, Ephem Spread)", fillcolor="#d1e7dd", color="#198754"];
    }

    // =========================================================================
    // STAGE 3: FEDERATED EDGE CLIENT PROCESSING
    // =========================================================================
    subgraph cluster_2 {
        label = "Stage 3: Federated Edge Client Processing (K = 5 Clients)";
        color = "#b02a37";
        style = "solid,rounded";
        bgcolor = "#fffbfc";
        fontcolor = "#842029";
        fontsize = 11.5;

        subgraph cluster_gat {
            label = "Domain-Specialized Client GAT Subgraphs (2-Layer, 8-Head Attention)";
            color = "#dc3545";
            style = "dotted,rounded";
            bgcolor = "#ffffff";
            fontcolor = "#842029";
            fontsize = 9.5;

            { rank=same; gat_temp; gat_cont; gat_behav; }

            gat_temp [label="Temporal GAT\\n(Relational Context)", fillcolor="#f8d7da", color="#dc3545"];
            gat_cont [label="Content GAT\\n(Relational Context)", fillcolor="#f8d7da", color="#dc3545"];
            gat_behav [label="Behavioral GAT\\n(Relational Context)", fillcolor="#f8d7da", color="#dc3545"];
        }

        subgraph cluster_opt {
            label = "Optimization & Topological Partitioning Engine";
            color = "#fd7e14";
            style = "dashed,rounded";
            bgcolor = "#fffdf7";
            fontcolor = "#b04a00";
            fontsize = 9.5;

            { rank=same; leiden; pool; loss_engine; }

            leiden [label="Leiden CPM Partitioning\\n(Well-Connected Guarantee)", fillcolor="#fff3cd", color="#ffc107"];
            pool [label="Centroid Pooling\\nUniform Mean (c_p)", fillcolor="#fff3cd", color="#ffc107"];
            loss_engine [label="Joint Regularization:\\nL_total = L_CB-Focal + mu * L_SupCon\\n(mu=0.1, tau=0.07, M=64, Beta=0.5)", fillcolor="#ffe5d0", color="#fd7e14", shape=note];

            leiden -> pool;
        }
    }

    // =========================================================================
    // STAGE 4: SERVER COORDINATION & MACRO OVERLAY
    // =========================================================================
    subgraph cluster_3 {
        label = "Stage 4: Server Coordination & Macro-Topology Construction";
        color = "#6f42c1";
        style = "dashed,rounded";
        bgcolor = "#faf8fd";
        fontcolor = "#432874";
        fontsize = 11;

        { rank=same; fedavg; server_sage; }

        fedavg [label="Edge-Count Weighted FedAvg Sync\\n(Federated Global Parameter Aggregation)", fillcolor="#e2d9f3", color="#6f42c1"];
        server_sage [label="GraphSAGE Server Macro-Overlay\\n(Top-5 Cosine Graph on Community Centroids)", fillcolor="#d0c2ee", color="#6f42c1"];
    }

    // =========================================================================
    // STAGE 5: ENSEMBLE DECISION ARBITRATION
    // =========================================================================
    subgraph cluster_4 {
        label = "Stage 5: Multi-Head Ensemble Decision Arbitration";
        color = "#055160";
        style = "solid,rounded";
        bgcolor = "#f0f9fb";
        fontcolor = "#055160";
        fontsize = 11;

        meta_concat [label="Meta-Feature Vector Fusion\\n[3x GAT Probabilities || Argmax || Server Macro-Embeddings || 24 Raw NetFlow Attributes]", fillcolor="#cff4fc", color="#0dcaf0"];
        rf_meta [label="Random Forest Meta-Classifier (100 Balanced Trees, Depth 20)", fillcolor="#9eeaf9", color="#055160", shape=box3d];
        pred_out [label="Final 8-Class Intrusion Detection Output\\n[Benign, Backdoor (99.57%), DDoS, DoS, Injection, Password, Scanning, XSS]", fillcolor="#198754", color="#0f5132", fontcolor="#ffffff", style="filled,bold", fontsize=10.5];

        meta_concat -> rf_meta -> pred_out;
    }

    // Pipeline Connections (Top to Bottom)
    sig_split -> feat_temp;
    sig_split -> feat_cont;
    sig_split -> feat_behav;

    feat_temp -> gat_temp;
    feat_cont -> gat_cont;
    feat_behav -> gat_behav;

    gat_temp -> leiden [color="#d63384"];
    gat_cont -> leiden [color="#d63384"];
    gat_behav -> leiden [color="#d63384"];

    loss_engine -> gat_temp [style=dotted, color="#fd7e14", label="Gradients"];
    loss_engine -> gat_cont [style=dotted, color="#fd7e14"];
    loss_engine -> gat_behav [style=dotted, color="#fd7e14"];

    gat_temp -> fedavg [style=dashed, dir=both, color="#6f42c1", label="FedAvg"];
    gat_cont -> fedavg [style=dashed, dir=both, color="#6f42c1"];
    gat_behav -> fedavg [style=dashed, dir=both, color="#6f42c1"];

    pool -> server_sage [label="Centroids", color="#6f42c1"];

    gat_temp -> meta_concat [label="Logits", color="#0dcaf0"];
    gat_cont -> meta_concat [color="#0dcaf0"];
    gat_behav -> meta_concat [color="#0dcaf0"];
    server_sage -> meta_concat [label="Macro Context", color="#6f42c1"];

    // Direct Skip Connection
    sig_split -> meta_concat [label="Statistical Skip Connection (Raw 24 Features) [+44.60% Bal Acc]", style=bold, color="#b02a37", penwidth=2.0];
}
"""

with open('papers/paper_latex/figures/architecture.dot', 'w', encoding='utf-8') as f:
    f.write(dot_code)

encoded = urllib.parse.quote(dot_code)
url = f"https://quickchart.io/graphviz?graph={encoded}&format=png"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as resp:
    img_data = resp.read()

png_path = 'papers/paper_latex/figures/fig1_architecture.png'
with open(png_path, 'wb') as f:
    f.write(img_data)

img = Image.open(io.BytesIO(img_data)).convert('RGB')
pdf_path = 'papers/paper_latex/figures/fig1_architecture.pdf'
img.save(pdf_path, 'PDF', resolution=300.0)
print(f"Rendered image size: {img.size}")
