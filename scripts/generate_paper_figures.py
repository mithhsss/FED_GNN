import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

os.makedirs('papers/paper_latex/figures', exist_ok=True)

# Set high-quality IEEE publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# =========================================================================
# FIGURE 1: Architectural Diagram
# =========================================================================
fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
ax.axis('off')

def draw_box(ax, xy, width, height, title, subtitle=None, facecolor='#f0f4f8', edgecolor='#1f4e79', lw=1.5, textcolor='#0b2238'):
    rect = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.03", 
                                  fc=facecolor, ec=edgecolor, lw=lw, zorder=2)
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    cy = xy[1] + height / 2.0
    if subtitle:
        ax.text(cx, cy + 0.12, title, ha='center', va='center', fontsize=9, fontweight='bold', color=textcolor, zorder=3)
        ax.text(cx, cy - 0.12, subtitle, ha='center', va='center', fontsize=7.5, color='#333333', zorder=3)
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color=textcolor, zorder=3)
    return rect

def draw_arrow(ax, start, end, label=None, style="->", color='#1f4e79', lw=1.5):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3), zorder=4)
    if label:
        mx = (start[0] + end[0]) / 2.0
        my = (start[1] + end[1]) / 2.0
        ax.text(mx, my + 0.08, label, ha='center', va='bottom', fontsize=7.5, fontweight='semibold', color='#0b2238', zorder=5)

# 1. Flow Ingestion & Grouping
draw_box(ax, (0.02, 0.55), 1.5, 0.8, "NF-ToN-IoT Flows", "8-Tuple Signature Grouping\n(Zero Evaluation Leakage)", facecolor='#e8f4fd', edgecolor='#0d6efd')
draw_box(ax, (0.02, 0.15), 1.5, 0.32, "Raw Flow Features", "24 Statistical Attributes", facecolor='#f8f9fa', edgecolor='#6c757d')

# 2. Client Specialized GATs
draw_box(ax, (1.9, 0.72), 1.8, 0.42, "Temporal GAT", "Inter-arrival & Duration", facecolor='#e2f0d9', edgecolor='#385723')
draw_box(ax, (1.9, 0.48), 1.8, 0.42, "Content GAT", "Payload Bytes & Flags", facecolor='#e2f0d9', edgecolor='#385723')
draw_box(ax, (1.9, 0.24), 1.8, 0.42, "Behavioral GAT", "Port Asymmetry & Spread", facecolor='#e2f0d9', edgecolor='#385723')

draw_arrow(ax, (1.52, 0.95), (1.9, 0.93))
draw_arrow(ax, (1.52, 0.75), (1.9, 0.69))
draw_arrow(ax, (1.52, 0.55), (1.9, 0.45))

# 3. Community Detection & Regularization Box
draw_box(ax, (4.1, 0.24), 1.7, 0.90, "Leiden Community\nPartitioning", "Guaranteed Connectivity\nModularity Optimization", facecolor='#fff2cc', edgecolor='#b25900')

draw_arrow(ax, (3.7, 0.93), (4.1, 0.85))
draw_arrow(ax, (3.7, 0.69), (4.1, 0.69))
draw_arrow(ax, (3.7, 0.45), (4.1, 0.53))

# Contrastive Loss Box
draw_box(ax, (2.6, 0.02), 2.7, 0.18, "L_total = L_CB-Focal + mu * L_SupCon (mu=0.1)", "Class-Balanced Sampling + Edge Interpolation", facecolor='#fce4d6', edgecolor='#c65911')

# 4. Federated Aggregation & Server Overlay
draw_box(ax, (6.2, 0.60), 1.8, 0.54, "GraphSAGE Server Overlay", "Top-k Cosine Similarity\nCommunity Centroid Graph", facecolor='#e8eaf6', edgecolor='#3949ab')
draw_box(ax, (6.2, 0.20), 1.8, 0.34, "FedAvg Aggregator", "Edge-Weighted Parameter Sync", facecolor='#ede7f6', edgecolor='#5e35b1')

draw_arrow(ax, (5.8, 0.69), (6.2, 0.87), label="Pooled Embs")
draw_arrow(ax, (5.8, 0.45), (6.2, 0.37), label="Local States")
draw_arrow(ax, (7.1, 0.20), (3.7, 0.20), style="<->", label="FedAvg Sync")

# 5. Meta-Classifier Fusion
draw_box(ax, (8.4, 0.35), 1.8, 0.65, "Random Forest Fusion", "Concat: 3x GAT Probs,\nServer Embs + Raw Feats\n(100 Trees, Balanced)", facecolor='#d1ecf1', edgecolor='#0c5460')

draw_arrow(ax, (8.0, 0.87), (8.4, 0.80), label="Server Logits")
draw_arrow(ax, (1.52, 0.31), (8.4, 0.45), label="Raw Features (Skip Connection)")

# Output
ax.text(10.35, 0.67, "Intrusion Output\n(8 Classes)", ha='left', va='center', fontsize=9, fontweight='bold', color='#0b2238')
draw_arrow(ax, (10.2, 0.67), (10.6, 0.67), color='#0d6efd')

ax.set_xlim(0, 10.8)
ax.set_ylim(-0.02, 1.25)
plt.tight_layout()
plt.savefig('papers/paper_latex/figures/fig1_architecture.pdf', bbox_inches='tight')
plt.savefig('papers/paper_latex/figures/fig1_architecture.png', bbox_inches='tight', dpi=300)
plt.close()
print("Fig 1 created.")

# =========================================================================
# FIGURE 2: Confusion Matrices (Seed 42)
# =========================================================================
classes = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']

# Seed 42 Ground Truth Confusion Matrices from architecture_and_results_report.md
cm_anchor = np.array([
    [52742,     0,     0,    0,     2,     0,     0,     0],
    [    0,     6,     0,    0,  3430,     0,     0,     0],
    [    0,     0, 58538,    0,  6893,     0,     0,     0],
    [    0,     0,     0, 3561,     5,     0,     0,     0],
    [    0,     0,     0,    0, 93572,     0,    10,     0],
    [    0,     0,     0,    0, 11610, 19650,     0,     0],
    [    0,     0,     0,    0,  1193,     0,  3064,     0],
    [    0,     0,     0,    0,     0,     0,     0, 19962]
], dtype=float)

cm_proposed = np.array([
    [52742,     0,     0,    0,     2,     0,     0,     0],
    [    0,  3419,     0,    0,    17,     0,     0,     0],
    [    0,     0, 58852,    0,  6579,     0,     0,     0],
    [    0,     0,     0, 3475,    91,     0,     0,     0],
    [    0,     0,     0,    0, 93576,     0,     6,     0],
    [    0,     0,     0,    0, 12587, 18673,     0,     0],
    [    0,     0,     0,    0,   252,     0,  4005,     0],
    [    0,     0,     0,    0,     0,     0,     0, 19962]
], dtype=float)

# Row-normalize to recall percentages
cm_anchor_norm = (cm_anchor / cm_anchor.sum(axis=1, keepdims=True)) * 100.0
cm_proposed_norm = (cm_proposed / cm_proposed.sum(axis=1, keepdims=True)) * 100.0

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

sns.heatmap(cm_anchor_norm, ax=axes[0], annot=True, fmt='.1f', cmap='Blues', cbar=False,
            xticklabels=classes, yticklabels=classes, vmin=0, vmax=100)
axes[0].set_title('(a) GNN Anchor (Plain CE, Seed 42)\nBackdoor Recall: 0.17% (3,430 misclassified into Injection)', fontsize=9.5, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontweight='semibold')
axes[0].set_ylabel('True Label', fontweight='semibold')
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(cm_proposed_norm, ax=axes[1], annot=True, fmt='.1f', cmap='Greens', cbar=True,
            xticklabels=classes, yticklabels=classes, vmin=0, vmax=100,
            cbar_kws={'label': 'Recall (%)'})
axes[1].set_title('(b) Proposed Model (mu=0.1 SupCon, Seed 42)\nBackdoor Recall: 99.51% (Only 17 misclassified into Injection)', fontsize=9.5, fontweight='bold', color='#1e5631')
axes[1].set_xlabel('Predicted Label', fontweight='semibold')
axes[1].set_ylabel('True Label', fontweight='semibold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('papers/paper_latex/figures/fig2_confusion_matrices.pdf', bbox_inches='tight')
plt.savefig('papers/paper_latex/figures/fig2_confusion_matrices.png', bbox_inches='tight', dpi=300)
plt.close()
print("Fig 2 created.")

# =========================================================================
# FIGURE 3: Per-Class Recalls & Balanced Accuracy across mu
# =========================================================================
# Multi-seed ground truth means and standard deviations from multi_seed_ablation_summary.json
configs = ['Anchor (CE)', 'mu=0.0', 'mu=0.1 (Prop.)', 'mu=0.3', 'mu=0.5']
x = np.arange(len(configs))

bal_acc_means = [78.11, 80.69, 94.28, 90.04, 86.04]
bal_acc_stds = [2.01, 1.64, 1.20, 6.95, 3.06]

backdoor_means = [0.11, 0.80, 99.57, 99.52, 99.08]
backdoor_stds = [0.08, 0.15, 0.16, 0.25, 0.40]

scanning_means = [80.80, 89.92, 97.98, 94.89, 91.03]
scanning_stds = [13.52, 9.36, 2.76, 4.83, 12.40]

password_means = [56.07, 68.63, 66.82, 70.30, 57.83]
password_stds = [5.38, 10.69, 5.19, 6.61, 2.88]

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=300)
width = 0.18

r1 = ax.bar(x - 1.5*width, bal_acc_means, width, yerr=bal_acc_stds, capsize=3, label='Balanced Accuracy', color='#1f77b4', edgecolor='black', lw=0.6)
r2 = ax.bar(x - 0.5*width, backdoor_means, width, yerr=backdoor_stds, capsize=3, label='Backdoor Recall', color='#d62728', edgecolor='black', lw=0.6)
r3 = ax.bar(x + 0.5*width, scanning_means, width, yerr=scanning_stds, capsize=3, label='Scanning Recall', color='#2ca02c', edgecolor='black', lw=0.6)
r4 = ax.bar(x + 1.5*width, password_means, width, yerr=password_stds, capsize=3, label='Password Recall', color='#ff7f0e', edgecolor='black', lw=0.6)

# Annotate the dramatic Backdoor jump
ax.annotate('Rescue: 0.11% -> 99.57%', xy=(2 - 0.5*width, 99.57), xytext=(1.0, 106),
            arrowprops=dict(arrowstyle="->", color='#b30000', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#b30000')

ax.set_ylabel('Metric (%)', fontweight='semibold')
ax.set_title('Performance Metrics Across SupCon Regularization Strength mu (Mean +/- Std, 3 Seeds)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontweight='semibold')
ax.set_ylim(0, 118)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower left', ncol=4, frameon=True)

plt.tight_layout()
plt.savefig('papers/paper_latex/figures/fig3_per_class_recalls.pdf', bbox_inches='tight')
plt.savefig('papers/paper_latex/figures/fig3_per_class_recalls.png', bbox_inches='tight', dpi=300)
plt.close()
print("Fig 3 created.")
