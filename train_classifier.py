"""
train_classifier.py — Train a classifier on SCOP categories using ESM-2 embeddings.

Loads the .pt embeddings, extracts SCOP labels from the FASTA file, trains
a configurable classifier (logistic regression or MLP), and reports accuracy,
per-class F1, and a confusion-matrix PNG.

Usage:
    python train_classifier.py \\
        --embeddings embeddings_esm2_650M.pt \\
        --fasta astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \\
        [--level class|fold|superfamily] \\
        [--classifier logreg|mlp] \\
        [--test-size 0.2] \\
        [--out-dir results/]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Label parsing (same logic as explore_embeddings.py)
# ---------------------------------------------------------------------------

def parse_scop_labels(fasta_path: str, level: str = "class") -> dict[str, str]:
    """Return {seq_id: scop_label} from ASTRAL SCOP FASTA headers.

    Header format:  >d1dlwa_ a.1.1.1 (A:) description...
    level choices:
        'class'       → 'a'
        'fold'        → 'a.1'
        'superfamily' → 'a.1.1'
        'family'      → 'a.1.1.1'
    """
    depth = {"class": 1, "fold": 2, "superfamily": 3, "family": 4}.get(level, 1)
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
            labels[seq_id] = ".".join(scop_fields[:depth])
    return labels


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    fig_size = max(8, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=(len(class_names) <= 20), fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=0.5, linecolor="grey",
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a SCOP-category classifier on ESM-2 embeddings."
    )
    p.add_argument("--embeddings", "-e", required=True,
                   help="Path to embeddings .pt file.")
    p.add_argument("--fasta", "-f", required=True,
                   help="Original FASTA file for SCOP label extraction.")
    p.add_argument("--level", default="class",
                   choices=["class", "fold", "superfamily", "family"],
                   help="SCOP hierarchy level to classify. Default: class.")
    p.add_argument("--classifier", "-c", default="mlp",
                   choices=["logreg", "mlp"],
                   help="Classifier type. Default: mlp.")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of data for testing. Default: 0.2.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--min-samples", type=int, default=5,
                   help="Minimum sequences per class (smaller classes removed). Default: 5.")
    p.add_argument("--out-dir", "-o", default="results",
                   help="Directory for output files. Default: results/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ──────────────────────────────────────────────────────
    print(f"Loading embeddings: {args.embeddings}")
    raw: dict[str, torch.Tensor] = torch.load(args.embeddings, map_location="cpu")

    ids = list(raw.keys())
    vectors = []
    for seq_id in ids:
        t = raw[seq_id]
        if t.dim() == 2:
            t = t.mean(dim=0)
        vectors.append(t.numpy())
    matrix = np.stack(vectors, axis=0)
    print(f"  {matrix.shape[0]} sequences × {matrix.shape[1]} dims")

    # ── Parse SCOP labels ────────────────────────────────────────────────────
    print(f"Parsing SCOP labels (level='{args.level}'): {args.fasta}")
    label_map = parse_scop_labels(args.fasta, level=args.level)
    raw_labels = [label_map.get(seq_id, "unknown") for seq_id in ids]

    # Count per class and filter rare ones
    from collections import Counter
    counts = Counter(raw_labels)
    valid_classes = {cls for cls, cnt in counts.items()
                     if cnt >= args.min_samples and cls != "unknown"}
    mask = [l in valid_classes for l in raw_labels]
    matrix = matrix[mask]
    labels_filtered = [l for l, m in zip(raw_labels, mask) if m]
    print(f"  {len(valid_classes)} classes with ≥{args.min_samples} samples "
          f"({sum(mask)} sequences kept)")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels_filtered)
    class_names = list(le.classes_)

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, y, test_size=args.test_size,
        random_state=args.random_seed, stratify=y,
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Standardise ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Build classifier ─────────────────────────────────────────────────────
    if args.classifier == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=args.random_seed,
                                 solver="lbfgs", multi_class="auto", n_jobs=-1)
        print("Training Logistic Regression …")
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=200,
            random_state=args.random_seed,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=True,
        )
        print("Training MLP classifier …")

    clf.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print(f"  SCOP level      : {args.level}")
    print(f"  Classifier      : {args.classifier}")
    print(f"  Test accuracy   : {acc * 100:.2f}%")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print("=" * 60)

    report = classification_report(y_test, y_pred, target_names=class_names,
                                   zero_division=0)
    print(report)

    # Save text report
    report_path = out_dir / f"report_{args.level}_{args.classifier}.txt"
    with open(report_path, "w") as fh:
        fh.write(f"SCOP level : {args.level}\n")
        fh.write(f"Classifier : {args.classifier}\n")
        fh.write(f"Test accuracy : {acc * 100:.2f}%\n")
        fh.write(f"Macro F1      : {macro_f1:.4f}\n\n")
        fh.write(report)
    print(f"  Report saved → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names,
                          out_dir / f"confusion_{args.level}_{args.classifier}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
