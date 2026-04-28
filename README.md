# seqBased_Embeddings

Generate sequence-based protein embeddings using **ESM-2** (Meta AI's protein language model), visualize them, and classify proteins by SCOP category.

---

## Repository Structure

```
.
├── astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt  ← ASTRAL SCOP 40% dataset
├── generate_embeddings.py    ← Step 1: run ESM-2 inference → .pt file
├── explore_embeddings.py     ← Step 2: PCA / t-SNE visualization
├── train_classifier.py       ← Step 3: train SCOP-class classifier
├── run.sh                    ← One-click script (GPU-ready)
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **ASTRAL SCOP FASTA** dataset (40% sequence identity, version 2.03):

```
astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt
```

12,119 protein sequences · SCOP class/fold/superfamily labels in each FASTA header.
The dataset file is included in this repository.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU note:** For faster inference install the CUDA-enabled torch build from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

---

## Step 1 — Generate Embeddings

```bash
# Quick start (one-click script, optimised for RTX 5090 / 24 GB+ GPU)
./run.sh

# Or manually:
python generate_embeddings.py \
    --input astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \
    --output embeddings_esm2_650M.pt \
    --model esm2_t33_650M_UR50D \
    --batch-size 32 \
    --per-sequence
```

### All `generate_embeddings.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | *(required)* | Path to input FASTA file |
| `--output` / `-o` | `embeddings.pt` | Output file (PyTorch `.pt` dict) |
| `--model` / `-m` | `esm2_t33_650M_UR50D` | ESM-2 model variant (see below) |
| `--batch-size` / `-b` | `8` | Sequences per forward pass (reduce if OOM) |
| `--repr-layer` | `-1` | Transformer layer to extract (`-1` = last layer) |
| `--per-sequence` | off | Mean-pool residues → one vector per sequence |
| `--max-len` | `1022` | Truncate sequences longer than this |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, etc. |

---

## Step 2 — Visualize Embeddings

```bash
python explore_embeddings.py \
    --embeddings embeddings_esm2_650M.pt \
    --fasta astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \
    --level class \
    --tsne \
    --out-dir plots/
```

Produces `plots/pca_class.png` (and `plots/tsne_class.png` with `--tsne`), coloured by SCOP level.

### `explore_embeddings.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--embeddings` / `-e` | *(required)* | Path to `.pt` embeddings |
| `--fasta` / `-f` | *(required)* | Original FASTA file |
| `--level` | `class` | SCOP level: `class`, `fold`, `superfamily`, `family` |
| `--tsne` | off | Also generate t-SNE plot |
| `--tsne-perplexity` | `30` | t-SNE perplexity |
| `--pca-components` | `50` | PCA pre-reduction before t-SNE |
| `--out-dir` / `-o` | same dir as embeddings | Output directory for PNG files |

---

## Step 3 — Train a Classifier

```bash
python train_classifier.py \
    --embeddings embeddings_esm2_650M.pt \
    --fasta astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \
    --level class \
    --classifier mlp \
    --out-dir results/
```

Outputs a classification report and confusion-matrix PNG to `results/`.

### `train_classifier.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--embeddings` / `-e` | *(required)* | Path to `.pt` embeddings |
| `--fasta` / `-f` | *(required)* | Original FASTA file |
| `--level` | `class` | SCOP level to classify |
| `--classifier` / `-c` | `mlp` | `logreg` (Logistic Regression) or `mlp` (Neural Net) |
| `--test-size` | `0.2` | Test split fraction |
| `--min-samples` | `5` | Min sequences per class (rare classes filtered out) |
| `--out-dir` / `-o` | `results/` | Output directory |

---

## Available ESM-2 Models

| Model name | Parameters | Embedding dim | Notes |
|---|---|---|---|
| `esm2_t6_8M_UR50D` | 8 M | 320 | Fastest, CPU-friendly |
| `esm2_t12_35M_UR50D` | 35 M | 480 | Small, CPU-friendly |
| `esm2_t30_150M_UR50D` | 150 M | 640 | Medium |
| `esm2_t33_650M_UR50D` | 650 M | 1280 | **Recommended** |
| `esm2_t36_3B_UR50D` | 3 B | 2560 | GPU recommended |
| `esm2_t48_15B_UR50D` | 15 B | 5120 | Needs multi-GPU |

---

## Loading the Embeddings Directly

```python
import torch

embeddings = torch.load("embeddings_esm2_650M.pt")

# embeddings is a dict: { sequence_id (str) -> torch.Tensor }
# Per-sequence (--per-sequence flag): shape (embed_dim,)       e.g. (1280,)
# Per-residue  (default):             shape (seq_len, embed_dim)

for seq_id, emb in embeddings.items():
    print(seq_id, emb.shape)
```

---

## References

- ESM-2 paper: [Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574)
- [fair-esm GitHub](https://github.com/facebookresearch/esm)
- [ASTRAL SCOP database](https://scop.berkeley.edu/)
