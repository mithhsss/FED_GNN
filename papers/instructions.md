# Paper Writing Instructions — Antigravity

## Title

A Contrastive Hybrid Federated GNN Architecture with Leakage-Controlled
Evaluation for IoT Intrusion Detection

## Template

IEEEtran, `\documentclass[journal]{IEEEtran}` (Overleaf built-in). Two-column,
IEEE journal style. Target length: 12-13 pages excluding references.

## Scope — what to include

- Base paper: FedGATSage (Al Tfaily et al., Scientific Reports 2025) — replicated and audited
- Our pipeline: 3 specialized GATs, Leiden community detection, community pooling,
  GraphSAGE server overlay, FedAvg, Random Forest fusion
- Our contributions: CB-Focal loss, class-balanced batch sampler, edge interpolation,
  Supervised Contrastive Loss (SupCon, mu ablation), signature-grouped leakage-free
  evaluation split, corrected FNR metric
- Results: NF-ToN-IoT only, multi-seed (3 seeds minimum), mean +/- std reported everywhere

## Scope — explicitly EXCLUDED, do not mention as implemented

- NF-UNSW-NB15 (non-determinism unresolved — do not report these numbers)
- Mixture-of-Experts fusion (RF outperformed it; if mentioned at all, it goes in a
  short "we also tested MoE and found RF superior" footnote/appendix only, not a
  main section, and only if space allows)
- Attention pooling, learned differentiable adjacency, DP — never implemented, omit entirely
  or one sentence in future work only, no formulas

## Citation rules (IEEE style)

1. Every citation must be verified against the actual paper before use — no citing
   from memory, no citing a paper whose title/venue/year you have not confirmed.
2. Numbered IEEE style: [1], [2], ... in order of first appearance, not alphabetical.
3. Every claim taken from a source paper must be paraphrased in your own words.
   NEVER copy a sentence, phrase, or formula description verbatim from any source PDF.
   Exception: mathematical formulas/equations may be reproduced exactly (equations
   are not prose plagiarism), but must be numbered, attributed to the source paper
   in the surrounding text, and use the paper's own notation only where necessary
   for correctness — otherwise adapt notation to match this paper's conventions.
4. No sentence in this paper may match more than ~8 consecutive words from any
   source PDF. If a technical term or the source paper's own name for a method
   requires exact wording (e.g. a named loss function), quote it in quotation
   marks with the citation immediately after.
5. Every number reported (accuracy, F1, FNR, hyperparameter values) must trace to
   an actual result JSON file in this project or an actual number stated in the
   cited paper — never invent, round misleadingly, or state a number "from memory."
6. Cross-check every citation's claimed venue/year against what was actually
   verified in this project's citation-check history before using it. If a paper's
   metadata (venue, year, page numbers) has not been independently verified,
   flag it in your response to me rather than using it silently.
7. Do not cite a source for a claim that source does not actually make (this
   project has previously caught multiple cases of "real paper, wrong formula
   attributed to it" — do not repeat this).

##
