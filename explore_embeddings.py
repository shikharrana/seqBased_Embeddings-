"""
explore_embeddings.py — Visualize ESM-2 embeddings and SCOP labels.

Loads the .pt embeddings produced by generate_embeddings.py, parses SCOP
class/fold/superfamily labels from the original FASTA, and produces:
  • PCA scatter plot  (always)
  • t-SNE scatter plot (--tsne flag)

Output PNG files are saved alongside the embeddings file by default.

Usage:
    python explore_embeddings.py \\
        --embeddings embeddings_esm2_650M.pt \\
        --fasta astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \\
        [--level class|fold|superfamily] \\
        [--tsne] \\
        [--out-dir plots/]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_scop_labels(fasta_path: str, level: str = "class") -> dict[str, str]:
    """Return {seq_id: scop_label} extracted from the FASTA header lines.

    ASTRAL SCOP headers look like:
        >d1dlwa_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin ...
    The second field is the hierarchical SCOP ID: class.fold.superfamily.family

    level choices:
        'class'       → 'a'
        'fold'        → 'a.1'
        'superfamily' → 'a.1.1'
        'family'      → 'a.1.1.1'
    """
    level_depth = {"class": 1, "fold": 2, "superfamily": 3, "family": 4}
    depth = level_depth.get(level, 1)

    labels: dict[str, str] = {}
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            parts = line[1:].split()
            if len(parts) < 2:
                continue
            seq_id = parts[0]
            scop_fields = parts[1].split(".")
            label = ".".join(scop_fields[:depth])
            labels[seq_id] = label
    return labels


# ---------------------------------------------------------------------------
# Colour palette helper
# ---------------------------------------------------------------------------

def build_palette(unique_labels: list[str]) -> dict[str, tuple]:
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    return {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter_plot(coords: np.ndarray, labels: list[str], title: str, out_path: Path):
    unique = sorted(set(labels))
    palette = build_palette(unique)
    colours = [palette[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colours, s=6, alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Legend (skip if too many classes)
    if len(unique) <= 30:
        patches = [mpatches.Patch(color=palette[l], label=l) for l in unique]
        ax.legend(handles=patches, fontsize=7, markerscale=1.5,
                  loc="best", ncol=max(1, len(unique) // 10))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize ESM-2 protein embeddings.")
    p.add_argument("--embeddings", "-e", required=True,
                   help="Path to embeddings .pt file (output of generate_embeddings.py).")
    p.add_argument("--fasta", "-f", required=True,
                   help="Original FASTA file (for extracting SCOP labels).")
    p.add_argument("--level", default="class",
                   choices=["class", "fold", "superfamily", "family"],
                   help="SCOP hierarchy level to colour by. Default: class.")
    p.add_argument("--tsne", action="store_true",
                   help="Also produce a t-SNE plot (slower).")
    p.add_argument("--tsne-perplexity", type=float, default=30.0,
                   help="t-SNE perplexity. Default: 30.")
    p.add_argument("--pca-components", type=int, default=50,
                   help="PCA components before t-SNE (speeds up t-SNE). Default: 50.")
    p.add_argument("--out-dir", "-o", default=None,
                   help="Directory for output plots. Default: same directory as --embeddings.")
    return p.parse_args()


def main():
    args = parse_args()

    emb_path = Path(args.embeddings)
    out_dir = Path(args.out_dir) if args.out_dir else emb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from: {emb_path}")
    raw: dict[str, torch.Tensor] = torch.load(str(emb_path), map_location="cpu")

    # If per-residue, mean-pool to get one vector per sequence
    ids = list(raw.keys())
    vectors = []
    for seq_id in ids:
        t = raw[seq_id]
        if t.dim() == 2:
            t = t.mean(dim=0)
        vectors.append(t.numpy())
    matrix = np.stack(vectors, axis=0)  # (N, D)
    print(f"  {matrix.shape[0]} sequences × {matrix.shape[1]} dims")

    # Parse SCOP labels
    print(f"Parsing SCOP labels (level='{args.level}') from: {args.fasta}")
    label_map = parse_scop_labels(args.fasta, level=args.level)
    labels = [label_map.get(seq_id, "unknown") for seq_id in ids]
    unique_labels = sorted(set(labels))
    print(f"  {len(unique_labels)} unique SCOP {args.level} labels")

    # PCA to 2D
    print("Running PCA …")
    n_components_2d = 2
    pca2 = PCA(n_components=n_components_2d, random_state=42)
    pca2d = pca2.fit_transform(matrix)
    var_explained = pca2.explained_variance_ratio_.sum() * 100
    title_pca = (f"PCA of ESM-2 embeddings  |  coloured by SCOP {args.level}\n"
                 f"({matrix.shape[0]} sequences, {var_explained:.1f}% variance explained)")
    scatter_plot(pca2d, labels, title_pca,
                 out_dir / f"pca_{args.level}.png")

    # t-SNE (optional)
    if args.tsne:
        print("Running t-SNE (this may take a few minutes) …")
        # Reduce to pca_components first for speed
        n_pre = min(args.pca_components, matrix.shape[1], matrix.shape[0])
        pre_pca = PCA(n_components=n_pre, random_state=42).fit_transform(matrix)
        perp = min(args.tsne_perplexity, matrix.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    n_iter=1000, verbose=1)
        tsne2d = tsne.fit_transform(pre_pca)
        title_tsne = (f"t-SNE of ESM-2 embeddings  |  coloured by SCOP {args.level}\n"
                      f"({matrix.shape[0]} sequences, perplexity={perp:.0f})")
        scatter_plot(tsne2d, labels, title_tsne,
                     out_dir / f"tsne_{args.level}.png")

    print("Done.")


if __name__ == "__main__":
    main()
